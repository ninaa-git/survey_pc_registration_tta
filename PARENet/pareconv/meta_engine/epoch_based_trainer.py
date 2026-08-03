import os
import os.path as osp
from typing import Tuple, Dict

import torch
import tqdm
import wandb

from pareconv.meta_engine.base_trainer import BaseTrainer
from pareconv.utils.torch import to_cuda
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.common import get_log_string
from pareconv.utils.data import precompute_neibors

from torch.utils.tensorboard import SummaryWriter


class EpochBasedTrainer(BaseTrainer):
    """
    Epoch-based trainer for the Point-TTA pipeline against the *batched*
    primary model.

    The previous version of this class buffered ``grad_acc_steps`` raw
    samples per virtual batch to (1) feed K aux inner steps and (2)
    accumulate the primary loss across samples. With the model now
    producing B pairs per ``data_dict``, the buffer-based virtual batching
    is gone:

        - JOINT phase: one forward / backward / step per ``data_dict``.
        - META  phase: ``meta_train_step`` iterates over the B pairs
                       *inside* the data_dict and performs the outer
                       optimiser step itself before returning.
    """

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
        self.writer = SummaryWriter()

    # ------------------------------------------------------------------
    # Hooks — override in subclass as needed
    # ------------------------------------------------------------------
    def before_train_step(self, epoch, iteration, data_dict) -> None: pass
    def before_val_step(self, epoch, iteration, data_dict) -> None: pass
    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None: pass
    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None: pass
    def before_train_epoch(self, epoch) -> None: pass
    def before_val_epoch(self, epoch) -> None: pass
    def after_train_epoch(self, epoch) -> None: pass
    def after_val_epoch(self, epoch) -> None: pass
    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]: pass
    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]: pass
    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None: pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data and model snapshot saved.')

    # ------------------------------------------------------------------
    # Helper: load + preprocess one raw batch from the DataLoader iterator
    # ------------------------------------------------------------------
    def _prepare_batch(self, raw_dict: dict) -> dict:
        data_dict = to_cuda(raw_dict)
        data = precompute_neibors(
            data_dict['points'],
            data_dict['lengths'],
            self.cfg.backbone.num_stages,
            self.cfg.backbone.num_neighbors,
        )
        data_dict.update(data)
        return data_dict

    # ------------------------------------------------------------------
    # train_epoch
    # ------------------------------------------------------------------
    def train_epoch(self):
        self.before_train_epoch(self.epoch)

        # Zero-grad at epoch start — accumulators are clean.
        self.optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()

        total_iterations = len(self.train_loader)

        for iteration, raw_dict in enumerate(self.train_loader):
            self.inner_iteration = iteration + 1
            self.iteration += 1

            data_dict = self._prepare_batch(raw_dict)

            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()

            if self.train_phase == 'meta': # meta phase
                output_dict, result_dict = self.meta_train_step(
                    self.epoch, self.inner_iteration, data_dict
                )

            else:  # joint phase
                output_dict, result_dict = self.joint_train_step(
                    self.epoch, self.inner_iteration, data_dict)

                is_last = (self.inner_iteration == total_iterations)
                if (self.inner_iteration % self.grad_acc_steps == 0) or is_last:
                    self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
                    self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # ── Post-step bookkeeping ────────────────────────────────────
            self.timer.add_process_time()

            self.after_train_step(
                self.epoch, self.inner_iteration, data_dict,
                output_dict, result_dict
            )

            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)

            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                for key, value in summary_dict.items():
                    self.writer.add_scalar(f'train_epoch/{key}', value, self.iteration)
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
                summary_dict_with_epoch['epoch']         = self.epoch
                summary_dict_with_epoch['learning_rate'] = self.get_lr()
                self.write_event('train', summary_dict_with_epoch, self.iteration)

            torch.cuda.empty_cache()

        # ── End of epoch ─────────────────────────────────────────────────
        self.after_train_epoch(self.epoch)
        self.logger.critical(
            get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        )

        if self.scheduler is not None:
            self.scheduler.step()

        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if not self.save_all_snapshots:
            last_snapshot = osp.join(self.snapshot_dir, f'epoch-{self.epoch - 1}.pth.tar')
            if osp.exists(last_snapshot):
                os.remove(last_snapshot)

    # ------------------------------------------------------------------
    # inference_epoch
    # ------------------------------------------------------------------
    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board  = SummaryBoard(adaptive=True)
        timer          = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations, ncols=180)

        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = self._prepare_batch(data_dict)

            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()

            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()

            self.after_val_step(
                self.epoch, self.inner_iteration, data_dict, output_dict, result_dict
            )

            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)

            pbar.set_description(get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            ))
            torch.cuda.empty_cache()

        self.after_val_epoch(self.epoch)

        summary_dict = summary_board.summary()
        for key, value in summary_dict.items():
            self.writer.add_scalar(f'val/{key}', value, self.iteration)
        self.logger.critical('[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer))

        summary_dict_with_epoch = summary_dict.copy()
        summary_dict_with_epoch['epoch'] = self.epoch
        self.write_event('val', summary_dict_with_epoch, self.iteration)

        self.set_train_mode()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            # --resume alone         → interrupted run in the current snapshot_dir
            # --resume --snapshot p  → explicit path, e.g. after moving files
            # Both cases: full restore (epoch + iteration + optimizers)
            snapshot_path = (
                self.args.snapshot
                if self.args.snapshot is not None
                else osp.join(self.snapshot_dir, 'snapshot.pth.tar')
            )
            self.load_snapshot(snapshot_path, resume=True)
        elif self.args.snapshot is not None:
            # Weights-only init — e.g. starting meta from a joint checkpoint.
            # epoch resets to 0, optimizers stay fresh.
            self.load_snapshot(self.args.snapshot, resume=False)

        self.set_train_mode()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()

        self.writer.close()
        if self.local_rank == 0:
            wandb.finish()