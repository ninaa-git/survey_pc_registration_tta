import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "LOCAL_RANK" in os.environ:
    import sys
    if not any(arg.startswith("--local_rank") for arg in sys.argv):
        sys.argv.append(f"--local_rank={os.environ['LOCAL_RANK']}")
import os.path as osp
import time
from collections import OrderedDict

import torch
import torch.optim as optim
import wandb

from pareconv.meta_engine.epoch_based_trainer import EpochBasedTrainer
from pareconv.utils.common import print_model_parameters

from config import make_cfg
from dataset import train_valid_data_loader
from tta.Source_Only.model import create_model
from loss import OverallLoss, Evaluator
from tta.Point_TTA.rec_aux import RecAux, RecAuxLoss
from common_utils.parser import make_pointtta_training_parser


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(
            cfg,
            parser=make_pointtta_training_parser(),
            max_epoch=cfg.optim.max_epoch,
            run_grad_check=False,
            autograd_anomaly_detection=False,
        )
        self.train_phase = self.args.train_phase
        self.logger.info(f'[Trainer] train_phase={self.train_phase}')

        torch.set_float32_matmul_precision('high')
        # ── Data ──────────────────────────────────────────────────────────
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        self.logger.info('Data loader created: {:.3f}s collapsed.'.format(time.time() - start_time))
        self.logger.info('Calibrate neighbors: {}.'.format(neighbor_limits))
        self.register_loader(train_loader, val_loader)

        # ── Main model ────────────────────────────────────────────────────
        model = create_model(cfg).cuda()
        model = self.register_model(model)

        """ if self.distributed:
            model.module.backbone    = torch.compile(model.module.backbone,    mode="default", dynamic=True)
        else:
            model.backbone    = torch.compile(model.backbone,    mode="reduce-overhead", dynamic=True)
            model.transformer = torch.compile(model.transformer,    mode="default", dynamic=True)
        """
        if self.local_rank == 0:
            print_model_parameters(model)

        # ── Aux head ─────────────────────────────────────────────────────
        self.aux_branch = RecAux(model, cfg).cuda()
        self.aux_loss   = RecAuxLoss().cuda()
        aux_params = [
            p for name, p in self.aux_branch.named_parameters()
            if not name.startswith('backbone.') and p.requires_grad
        ]
        print_model_parameters(self.aux_branch)

        main_params  = list(filter(lambda p: p.requires_grad, model.parameters()))

        # ── Optimizers ────────────────────────────────────────────────────
        if self.train_phase == 'joint':
            joint_optimizer = optim.Adam(
                main_params + aux_params,
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )
            self.register_optimizer(joint_optimizer)

            scheduler = optim.lr_scheduler.StepLR(joint_optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
            self.register_scheduler(scheduler)

        else:  # meta phase
            # Outer: Adam on θ_shar + θ_pri (β = meta_lr).
            outer_optimizer = optim.Adam(main_params, lr=self.args.meta_lr)
            self.register_optimizer(outer_optimizer)

            # Inner: SGD on θ_shar + θ_aux (α = meta_lr).
            self._shared_params = [
                p for n, p in self.aux_branch.named_parameters()
                if n.startswith('backbone.') and p.requires_grad
            ]
            self._aux_decoder_params = aux_params
            inner_params = self._shared_params + self._aux_decoder_params
            inner_optimizer = optim.SGD(inner_params, lr=self.args.meta_lr)
            self.register_aux_optimizer(inner_optimizer)

        # ── Misc ──────────────────────────────────────────────────────────
        # K = number of inner SGD steps on the auxiliary loss per outer
        # iteration (paper: 5 in total, applied to the *batched* aux loss
        # — not per pair).
        self.inner_steps = self.args.niter

        self.loss_func  = OverallLoss(cfg).cuda()
        self.evaluator  = Evaluator(cfg).cuda()

        self.best_metrics = float('inf')
        self.best_epoch   = 0

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _build_ckpt(self):
        """Common checkpoint keys for both phases."""
        model_state = self.model.state_dict()
        if self.distributed:
            model_state = OrderedDict([(k[7:], v) for k, v in model_state.items()])
        ckpt = {
            'epoch':     self.epoch,
            'iteration': self.iteration,
            'model':     model_state,
            'train_phase': self.train_phase,
        }
        # Aux branch weights (skip the shared backbone — already in 'model')
        aux_state = {k: v for k, v in self.aux_branch.state_dict().items()
             if not k.startswith('backbone.')}
        ckpt['rec_branch'] = aux_state
        return ckpt

    def save_snapshot(self, filename):
        """Override to include aux branches in every snapshot."""
        if self.local_rank != 0:
            return

        ckpt = self._build_ckpt()

        # Per-epoch file
        path = osp.join(self.snapshot_dir, filename)
        torch.save(ckpt, path)
        self.logger.info(f'Snapshot saved to "{path}"')

        # snapshot.pth.tar — full state for resuming
        ckpt['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            ckpt['scheduler'] = self.scheduler.state_dict()
        if self.aux_optimizer is not None:
            ckpt['aux_optimizer'] = self.aux_optimizer.state_dict()
        snap_path = osp.join(self.snapshot_dir, 'snapshot.pth.tar')
        torch.save(ckpt, snap_path)
        self.logger.info(f'Snapshot saved to "{snap_path}"')

    # ------------------------------------------------------------------
    # Joint train step (joint phase)
    # ------------------------------------------------------------------
    # The model, aux_branch, and aux_loss are all batched-aware:
    # one forward pass on the full batched data_dict; aux_loss averages
    # the chamfer over all 2B reconstructed clouds. No adaptation.
    # ------------------------------------------------------------------
    def joint_train_step(self, epoch, iteration, data_dict):
        # Aux forward: returns a length-B list of (rec_ref, rec_src);
        # aux_loss reduces over all pairs (mean).
        self.aux_branch.train()
        recs = self.aux_branch(data_dict)
        loss_aux = self.aux_loss(recs, data_dict)

        # Primary forward
        output_dict = self.model(data_dict)
        loss_main = self.loss_func(output_dict, data_dict)
        joint_loss = loss_main['loss'] + loss_aux

        result_dict = self.evaluator(output_dict, data_dict)
        result_dict['loss']      = joint_loss          # live tensor — used for .backward()
        result_dict['loss_main'] = loss_main['loss'].detach()
        result_dict['loss_aux']  = loss_aux.detach()
        result_dict['c_loss']    = loss_main['c_loss'].detach()
        result_dict['f_ri_loss'] = loss_main['f_ri_loss'].detach()
        result_dict['f_re_loss'] = loss_main['f_re_loss'].detach()
        return output_dict, result_dict

    # ------------------------------------------------------------------
    # Meta train step  (Stage 2)
    # ------------------------------------------------------------------
    # Algorithm 1, applied to the whole batched data_dict (B pairs at once).
    #
    # Outer: Adam on θ_shar + θ_pri  (β = meta_lr).
    # Inner: SGD  on θ_shar + θ_aux  (α = meta_lr).
    #
    # 1. K inner SGD steps on L_aux(data_dict) — the aux loss is averaged
    #    over the 2B reconstructed clouds, so all pairs contribute to each
    #    of the K updates. After the K steps, θ_shar has moved to ϕ_shar
    #    and θ_aux has moved persistently. NO snapshot, NO restore.
    #
    # 2. One outer Adam step on L_pri(data_dict) at the adapted weights.
    #    L_pri is the standard batched primary loss (Σ_b implicit in the
    #    loss function). This corresponds to Σ_b ∇_θ L_pri in Eq. 9 of
    #    the paper, applied directly to the current parameters.
    #
    # Paper note: Algorithm 1 line 9 writes ``θ ← θ − β · …`` as if θ
    # were the pre-adaptation snapshot. In this implementation we do not
    # restore — the inner update on θ_shar persists, and the outer step
    # lands on top of it. This is the simpler "no-restore" variant and
    # matches the way most meta-aux registration codebases run it.
    # ------------------------------------------------------------------
    def meta_train_step(self, epoch, iteration, data_dict):
        # ── 1. Inner loop: K SGD steps on the batched aux loss.
        self.aux_branch.train()
        aux_loss_pre = None
        for k in range(self.inner_steps):
            self.aux_optimizer.zero_grad(set_to_none=True)
            recs = self.aux_branch(data_dict)
            loss_aux = self.aux_loss(recs, data_dict)
            if k == 0:
                aux_loss_pre = loss_aux.detach()   # pre-adaptation, for logging
            loss_aux.backward()
            self.aux_optimizer.step()

        # ── 2. Outer step on the primary loss at the adapted weights.
        #     The aux_optimizer cleared its own params' grads; main params
        #     (θ_shar + θ_pri) may carry stale grads — zero explicitly.
        self.optimizer.zero_grad(set_to_none=True)

        output_dict = self.model(data_dict)
        loss_main = self.loss_func(output_dict, data_dict)
        loss_main['loss'].backward()

        self.after_backward(epoch, iteration, data_dict, output_dict, None)
        self.check_gradients(epoch, iteration, data_dict, output_dict, None)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # ── 3. Logging
        result_dict = self.evaluator(output_dict, data_dict)
        result_dict['loss']      = loss_main['loss'].detach()
        result_dict['loss_main'] = loss_main['loss'].detach()
        result_dict['loss_aux']  = aux_loss_pre
        result_dict['c_loss']    = loss_main['c_loss'].detach()
        result_dict['f_ri_loss'] = loss_main['f_ri_loss'].detach()
        result_dict['f_re_loss'] = loss_main['f_re_loss'].detach()
        return output_dict, result_dict

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_main = self.loss_func(output_dict, data_dict)

        result_dict = self.evaluator(output_dict, data_dict)
        result_dict['loss']      = loss_main['loss'].detach()
        result_dict['c_loss']    = loss_main['c_loss'].detach()
        result_dict['f_ri_loss'] = loss_main['f_ri_loss'].detach()
        result_dict['f_re_loss'] = loss_main['f_re_loss'].detach()
        return output_dict, result_dict

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def before_val_epoch(self, epoch):
        self._val_acc   = {}
        self._val_count = 0

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        self._val_count += 1
        for k, v in result_dict.items():
            try:
                fv = float(v)
                if math.isnan(fv):
                    continue
                self._val_acc[k] = self._val_acc.get(k, 0.0) + fv
            except (TypeError, ValueError):
                pass

    def after_val_epoch(self, epoch):
        n         = max(self._val_count, 1)
        means     = {k: v / n for k, v in self._val_acc.items()}
        mean_rmse = means.get('RMSE', float('inf'))

        if self.local_rank == 0:
            self.write_event('val', means, epoch)

        if mean_rmse < self.best_metrics:
            self.best_metrics = mean_rmse
            self.best_epoch   = epoch
            if self.local_rank == 0:
                path = osp.join(self.snapshot_dir, 'best_model.pth.tar')
                ckpt = self._build_ckpt()
                ckpt['best_rmse'] = self.best_metrics
                torch.save(ckpt, path)
                self.logger.info(
                    f'[Best] Epoch {epoch}  RMSE={self.best_metrics:.4f}  → {path}'
                )


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()