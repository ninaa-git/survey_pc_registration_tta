import pdb
import sys
import argparse
import os
import os.path as osp
import time
import json
import abc
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import wandb

from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.torch import all_reduce_tensors, release_cuda, initialize
from pareconv.meta_engine.logger import Logger


def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--snapshot', default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--log_steps', type=int, default=30)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--model_desc', type=str, default='default_configs')
    return parser


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        cfg,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=False,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()
        self.cfg = cfg
        self.debug = self.args.debug

        time_stamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = osp.join(cfg.log_dir, 'train-{}.log'.format(time_stamp))
        self.logger = Logger(log_file=log_file, local_rank=self.args.local_rank)

        self.logger.info('Command executed: ' + ' '.join(sys.argv))

        # cfg may not be JSON-serialisable — convert safely
        try:
            self.logger.info('Configs:\n' + json.dumps(dict(cfg), indent=4))
        except TypeError:
            self.logger.info('Configs:\n' + str(cfg))

        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')

        self.distributed = self.args.local_rank != -1
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.local_rank = self.args.local_rank
            self.logger.info(f'Using DistributedDataParallel mode (world_size: {self.world_size})')
        else:
            self.world_size = 1
            self.local_rank = 0
            self.logger.info('Using Single-GPU mode.')

        self.cudnn_deterministic = cudnn_deterministic
        self.autograd_anomaly_detection = autograd_anomaly_detection
        self.seed = cfg.seed + self.local_rank
        initialize(
            seed=self.seed,
            cudnn_deterministic=self.cudnn_deterministic,
            autograd_anomaly_detection=self.autograd_anomaly_detection,
        )

        self.snapshot_dir = os.path.join(cfg.snapshot_dir, time_stamp)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.log_steps = self.args.log_steps
        self.run_grad_check = run_grad_check
        self.save_all_snapshots = save_all_snapshots

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.aux_optimizer = None
        self.epoch = 0
        self.iteration = 0
        self.inner_iteration = 0

        self.train_loader = None
        self.val_loader = None
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)
        self.timer = Timer()
        self.saved_states = {}

        self.training = True
        self.grad_acc_steps = grad_acc_steps
        if not self.debug:
            if self.local_rank == 0:
                wandb.init(
                    project='PARE-Net-meta',
                    name=f'{self.args.train_phase}_{time_stamp}',
                )
                self.logger.info(f'Wandb is enabled. Write events to {cfg.event_dir}.')

    def save_snapshot(self, filename):
        if self.local_rank != 0:
            return

        model_state_dict = self.model.state_dict()
        if self.distributed:
            model_state_dict = OrderedDict([(k[7:], v) for k, v in model_state_dict.items()])

        # ── Base state dict — saved in BOTH the per-epoch file and snapshot ──
        state_dict = {
            'epoch':     self.epoch,
            'iteration': self.iteration,
            'model':     model_state_dict,
        }

        state_dict['rec_branch'] = self.aux_branch.state_dict()

        # ── Per-epoch file ────────────────────────────────────────────────
        epoch_filename = osp.join(self.snapshot_dir, filename)
        torch.save(state_dict, epoch_filename)
        self.logger.info(f'Model saved to "{epoch_filename}"')

        # ── snapshot.pth.tar — adds optimizer states on top ───────────────
        snapshot_filename = osp.join(self.snapshot_dir, 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        if self.aux_optimizer is not None:
            state_dict['aux_optimizer'] = self.aux_optimizer.state_dict()
        torch.save(state_dict, snapshot_filename)
        self.logger.info(f'Snapshot saved to "{snapshot_filename}"')

    def load_snapshot(self, snapshot, fix_prefix=True, resume=False):
        self.logger.info(
            f'Loading snapshot from "{snapshot}" (resume={resume}).'
        )
        ckpt = torch.load(snapshot, map_location=torch.device('cpu'))

        # ── 1. Main model ─────────────────────────────────────────────────
        model_dict = ckpt['model']
        if fix_prefix and self.distributed:
            model_dict = OrderedDict([('module.' + k, v) for k, v in model_dict.items()])
        self.model.load_state_dict(model_dict, strict=True)
        self.logger.info('Main model loaded.')

        # ── 2. Aux branch ─────────────────────────────────────────────
        if 'rec_branch' in ckpt:
            aux_state = {k: v for k, v in ckpt['rec_branch'].items()
                        if not k.startswith('backbone.')}
            self.aux_branch.load_state_dict(aux_state, strict=False)
            self.logger.info('aux_branch (rec_branch) loaded.')
        else:
            self.logger.warning('No rec_branch in checkpoint — aux_branch keeps random init.')

        # ── 4. Training state — only when resuming a crashed run ──────────
        if resume:
            if 'epoch' in ckpt:
                self.epoch = ckpt['epoch']
                self.logger.info(f'Epoch restored: {self.epoch}')
            if 'iteration' in ckpt:
                self.iteration = ckpt['iteration']
                self.logger.info(f'Iteration restored: {self.iteration}')
            if 'optimizer' in ckpt and self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.logger.info('Optimizer loaded.')
            if 'scheduler' in ckpt and self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt['scheduler'])
                self.logger.info('Scheduler loaded.')
            if 'aux_optimizer' in ckpt and self.aux_optimizer is not None:
                self.aux_optimizer.load_state_dict(ckpt['aux_optimizer'])
                self.logger.info('Aux optimizer loaded.')
        else:
            self.logger.info(
                'Weights-only load — epoch/iteration/optimizers intentionally not restored.'
            )

    def register_model(self, model):
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        self.model = model
        self.logger.info('Model description:\n' + str(model))
        return model

    def register_optimizer(self, optimizer):
        if self.distributed:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * self.world_size
        self.optimizer = optimizer

    def register_aux_optimizer(self, aux_optimizer):
        self.aux_optimizer = aux_optimizer

    def register_scheduler(self, scheduler):
        self.scheduler = scheduler

    def register_loader(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def optimizer_step(self, iteration):
        if iteration % self.grad_acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def aux_optimizer_step(self, iteration):
        self.aux_optimizer.step()
        self.aux_optimizer.zero_grad()

    def save_state(self, key, value):
        self.saved_states[key] = release_cuda(value)

    def read_state(self, key):
        return self.saved_states[key]

    def check_invalid_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any():
                self.logger.error(f'NaN gradient in {name}.')
                return False
            if torch.isinf(param.grad).any():
                self.logger.error(f'Inf gradient in {name}.')
                return False
        return True

    def release_tensors(self, result_dict):
        if self.distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        result_dict = release_cuda(result_dict)
        return result_dict

    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def write_event(self, phase, event_dict, index):
        r"""Write wandb event."""
        if self.local_rank != 0:
            return
        new_dict = {}
        for key, value in event_dict.items():
            new_dict[f'{phase}/{key}'] = value
        if not self.debug:
            wandb.log(new_dict, step=index)

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError