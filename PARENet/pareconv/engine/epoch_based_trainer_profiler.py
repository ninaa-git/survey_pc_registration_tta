import os
import os.path as osp
import pdb
from typing import Tuple, Dict

try:
    import ipdb
except ImportError:
    pass
import torch
import torch.distributed as dist
import tqdm

from pareconv.engine.base_trainer import BaseTrainer
from pareconv.utils.torch import to_cuda
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.common import get_log_string
from pareconv.utils.data import precompute_subsample, precompute_neibors

from torch.utils.tensorboard import SummaryWriter

import wandb
from torch.profiler import profile, ProfilerActivity, record_function

import os.path as osp
from torch.profiler import (
    profile, ProfilerActivity, record_function,
    schedule, tensorboard_trace_handler,
)

import time

class EpochBasedTrainer_profiler(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_epoch,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=False,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_epoch = max_epoch
        if self.local_rank == 0:
            self.writer = SummaryWriter()
        else:
            self.writer = None

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        if self.distributed and self.train_loader is not None:
            sampler = getattr(self.train_loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

    def before_val_epoch(self, epoch) -> None:
        if self.distributed and self.val_loader is not None and len(self.val_loader) > 0:
            sampler = getattr(self.val_loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')

    def train_epoch(self):
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)

        # ---- profiler setup: only profile epoch 1 ----
        do_profile = (self.epoch == 1) and (self.local_rank == 0)
        trace_dir = osp.join(self.snapshot_dir, f"profiler_epoch{self.epoch}")
        prof_ctx = (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=20, active=5, repeat=1),
                on_trace_ready=tensorboard_trace_handler(trace_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,   # set True if you want source-line attribution (slower)
            )
            if do_profile else None
        )
        if prof_ctx is not None:
            prof_ctx.start()
        # ----------------------------------------------

        import time
        t_data = t_model = 0.0
        t = time.perf_counter()
        for iteration, data_dict in enumerate(self.train_loader):
            self.inner_iteration = iteration + 1
            self.iteration += 1
            data_dict = to_cuda(data_dict)
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()

            output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)

            if not torch.isfinite(result_dict['loss']):
                self.logger.warning(
                    f"Non-finite loss at epoch={self.epoch}, iter={self.inner_iteration}, "
                    f"loss={result_dict['loss'].item()}; sub-losses="
                    + ", ".join(f"{k}={v.item() if hasattr(v, 'item') else v}"
                                for k, v in result_dict.items() if k != 'loss')
                )
                self.timer.add_process_time()
                if self.distributed:
                    fake = sum((p.sum() * 0.0) for p in self.model.parameters() if p.requires_grad)
                    fake.backward()
                    self.optimizer.zero_grad()
                continue

            result_dict['loss'].backward()
            self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.inner_iteration)
            self.timer.add_process_time()
            self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            if self.local_rank == 0:
                self.summary_board.update_from_result_dict(result_dict)
                
                if self.inner_iteration % self.log_steps == 0:
                    summary_dict = self.summary_board.summary()
                    for key, value in summary_dict.items():
                        self.writer.add_scalar(f'epoch/{key}', value, self.epoch)
                    message = get_log_string(
                        result_dict=summary_dict,
                        epoch=self.epoch,
                        max_epoch=self.max_epoch,
                        iteration=self.inner_iteration,
                        max_iteration=total_iterations,
                        lr=self.get_lr(),
                        timer=self.timer,
                    )
                    self.logger.info(message)
                    summary_dict_with_epoch = summary_dict.copy()
                    summary_dict_with_epoch['epoch'] = self.epoch
                    summary_dict_with_epoch['learning_rate'] = self.get_lr()
                    self.write_event('train', summary_dict_with_epoch, self.iteration)

            # ---- profiler step ----
            if prof_ctx is not None:
                prof_ctx.step()
            # -----------------------

        # ---- finish profiler ----
        if prof_ctx is not None:
            prof_ctx.stop()
            # Print a quick summary in the log
            self.logger.info(
                prof_ctx.key_averages().table(
                    sort_by="cuda_time_total", row_limit=25
                )
            )
            # Upload the trace files to wandb as an artifact
            if not self.debug and self.local_rank == 0:
                artifact = wandb.Artifact(f"profiler_epoch{self.epoch}", type="profile")
                artifact.add_dir(trace_dir)
                wandb.log_artifact(artifact)
        # -------------------------

        self.after_train_epoch(self.epoch)
        if self.local_rank == 0:
            message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
            self.logger.critical(message)
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if self.local_rank == 0 and not self.save_all_snapshots:
            last_snapshot = osp.join(self.snapshot_dir, f'epoch-{self.epoch - 1}.pth.tar')
            if osp.exists(last_snapshot):
                os.remove(last_snapshot)
        if self.distributed:
            dist.barrier()

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)

        if len(self.val_loader) == 0:
            self.logger.warning('[Val] Validation loader is empty, skipping inference epoch.')
            self.set_train_mode()
            return
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        if self.local_rank == 0:
            pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations, ncols=180)
        else:
            pbar = enumerate(self.val_loader)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_process_time()
            self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            if self.local_rank == 0:
                message = get_log_string(
                    result_dict=summary_board.summary(),
                    epoch=self.epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    timer=timer,
                )
                pbar.set_description(message)
        self.after_val_epoch(self.epoch)
        summary_dict = summary_board.summary()
        if self.local_rank == 0 and self.writer is not None:
            for key, value in summary_dict.items():
                self.writer.add_scalar(f'val/{key}', value, self.epoch)
            message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
            self.logger.critical(message)

            if not hasattr(self, 'best_metric') or summary_dict.get('RMSE', float('inf')) < self.best_metric:
                self.best_metric = summary_dict.get('RMSE', float('inf'))
                self.save_best_snapshot('best.pth.tar')
                
            summary_dict_with_epoch = summary_dict.copy()
            summary_dict_with_epoch['epoch'] = self.epoch
       
            self.write_event('val', summary_dict_with_epoch, self.iteration)
        
        if self.distributed:
            dist.barrier()
        self.set_train_mode()

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()
        # self.inference_epoch()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()
        if not self.debug and self.local_rank == 0:
            if self.writer is not None:
                self.writer.close()
            wandb.finish()

        if self.distributed:
            dist.destroy_process_group()