"""
single_tester.py

  SingleTester        Standard inference: model.eval() + no_grad.
  SingleTesterTTA     Algorithm 1 test-time adaptation (Point-TTA style).

The TTA inner loop must mirror training:
  - SGD optimizer (no momentum), lr = α
  - K = self.tta_n_steps  (paper: 5)
  - Adapts BACKBONE + AUX DECODERS together (matches the inner_params built
    in trainval.py meta phase). Registration heads are frozen.
  - Episodic: backbone AND decoders snapshotted before each test pair and
    restored after, so adaptation does not leak between samples.
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from pareconv.meta_engine.base_tester import BaseTester
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.common import get_log_string
from pareconv.utils.torch import release_cuda, to_cuda
from pareconv.utils.data import precompute_neibors

from rec_aux import RecAuxLoss

import time
import numpy as np
from torch.profiler import profile, ProfilerActivity


# Names of the registration head submodules on `self.model`. These are
# eval()'d and have requires_grad disabled during adaptation. L_aux has no
# graph path through them so this is mostly hygiene, but keeping their
# requires_grad=False ensures the inner optimizer can never see them even if
# someone later adds them to the param list.
_FROZEN_HEADS = (
    'transformer', 'coarse_matching',
    'fine_matching', 'point_matching', 'coarse_target',
)


def _prep(data_dict, cfg):
    data_dict = to_cuda(data_dict)
    data = precompute_neibors(
        data_dict['points'], data_dict['lengths'],
        cfg.backbone.num_stages, cfg.backbone.num_neighbors,
    )
    data_dict.update(data)
    return data_dict


# ---------------------------------------------------------------------------
# Standard tester
# ---------------------------------------------------------------------------

class SingleTester(BaseTester):
    """Standard evaluation — model.eval() + torch.no_grad()."""

    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self): pass
    def before_test_step(self, iteration, data_dict): pass
    def test_step(self, iteration, data_dict) -> Dict: pass
    def eval_step(self, iteration, data_dict, output_dict) -> Dict: pass
    def after_test_step(self, iteration, data_dict, output_dict, result_dict): pass
    def after_test_epoch(self): pass
    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        if getattr(self.args, 'meta_snapshot', None) is not None:
            self.load_meta_snapshot(self.args.meta_snapshot, aux_branch=None)
        else:
            self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        pbar = tqdm(enumerate(self.test_loader),
                    total=len(self.test_loader), ncols=180)
        for iteration, data_dict in pbar:
            torch.cuda.synchronize(); timer.add_prepare_time()
            self.iteration = iteration + 1
            data_dict = _prep(data_dict, self.cfg)
            self.before_test_step(self.iteration, data_dict)
            torch.cuda.synchronize(); timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize(); timer.add_process_time()
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            self.logger.critical(
                self.summary_string(self.iteration, data_dict, output_dict, result_dict)
                + f', {timer.tostring()}'
            )
            torch.cuda.empty_cache()
        self.after_test_epoch()
        self.logger.critical(
            get_log_string(result_dict=summary_board.summary(), timer=timer)
        )


# ---------------------------------------------------------------------------
# TTA tester — Algorithm 1
# ---------------------------------------------------------------------------

class SingleTesterTTA(BaseTester):
    """
    Test-time adaptation tester — Point-TTA Algorithm 1, Eq. 10.

    Per test pair (P, Q):
        snapshot θ                          # backbone + decoders
        for t = 0 … K-1:
            loss = L_aux(aux_branch(P, Q))
            SGD step with lr=α on (backbone + decoders)
        run primary forward with adapted backbone
        evaluate
        restore θ                           # episodic reset
    """

    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

        # cfg.tta is populated by TesterTTA.__init__ from CLI args.
        tta = getattr(cfg, 'tta', None)
        def _g(k, d): return getattr(tta, k, d) if tta is not None else d

        self.tta_lr      = _g('lr',     2.5e-5)
        self.tta_n_steps = _g('niter',  5)

        self.rec_loss = RecAuxLoss()

        self.aux_branch:         Optional[nn.Module] = None
        self._tta_optimizer:     Optional[torch.optim.Optimizer] = None
        self._backbone_snapshot: Optional[dict] = None
        self._decoder_snapshot:  Optional[dict] = None

    def before_test_epoch(self): pass
    def before_test_step(self, iteration, data_dict): pass
    def test_step(self, iteration, data_dict) -> Dict: pass
    def eval_step(self, iteration, data_dict, output_dict) -> Dict: pass
    def after_test_step(self, iteration, data_dict, output_dict, result_dict): pass
    def after_test_epoch(self): pass
    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    # ---------------------------------------------------- registration
    def register_aux(self, aux_branch: nn.Module):
        """
        Register the RecAux head and build the SGD inner optimizer.
        Must be called after register_model().

        The optimizer's param list mirrors trainval.py meta phase
        `inner_params = self._shared_params + self._aux_decoder_params`:
        all aux_branch parameters that require grad — that's backbone (shared
        with self.model.backbone) plus the two decoders.
        """
        assert self.model is not None, 'Call register_model() first.'
        self.aux_branch = aux_branch

        # All aux_branch params: backbone.* (aliased to model.backbone) +
        # decoder_ref.* + decoder_src.*. Same set the training inner loop steps.
        inner_params = [p for p in self.aux_branch.parameters() if p.requires_grad]

        # SGD without momentum — must match training inner optimizer so the
        # meta-trained θ is being adapted with the same update rule it was
        # trained against.
        self._tta_optimizer = torch.optim.SGD(inner_params, lr=self.tta_lr)
        self.logger.info(
            f'[TTA] {len(inner_params)} adaptable params (backbone + decoders), '
            f'optimizer=SGD, lr={self.tta_lr}, K={self.tta_n_steps}'
        )

    # ---------------------------------------------------- checkpoints
    def _load_checkpoints(self):
        assert self.args.meta_snapshot is not None, \
            '--meta_snapshot is required for TTA mode.'
        self.load_meta_snapshot(self.args.meta_snapshot, aux_branch=self.aux_branch)

    # ---------------------------------------------------- mode helpers
    def _set_adapt_mode(self):
        """
        Adaptable: backbone + aux decoders.
        Frozen:    registration heads.
        """
        self.model.train()
        # Freeze registration heads — their grads would be undefined under L_aux
        # anyway (no graph path), but disabling requires_grad keeps the inner
        # optimizer from ever touching them if its param list is rebuilt.
        for name in _FROZEN_HEADS:
            m = getattr(self.model, name, None)
            if m is not None:
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

        # aux_branch.backbone IS model.backbone — already covered.
        # Decoders need requires_grad=True for the inner SGD step.
        if self.aux_branch is not None:
            self.aux_branch.train()
            for p in self.aux_branch.parameters():
                p.requires_grad_(True)

    def _set_eval_mode(self):
        """Restore eval mode for the primary forward, re-enable head grads."""
        self.model.eval()
        for name in _FROZEN_HEADS:
            m = getattr(self.model, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    def _save_snapshot(self):
        """Snapshot every parameter that the inner loop can move."""
        self._backbone_snapshot = copy.deepcopy(self.model.backbone.state_dict())
        # Decoder weights only — backbone.* in aux_branch.state_dict() is the
        # same tensors as model.backbone, no need to copy twice.
        if self.aux_branch is not None:
            self._decoder_snapshot = {
                k: v.detach().clone()
                for k, v in self.aux_branch.state_dict().items()
                if not k.startswith('backbone.')
            }

    def _restore_snapshot(self):
        """Episodic reset: undo the K SGD steps before the next test pair."""
        if self._backbone_snapshot is not None:
            self.model.backbone.load_state_dict(self._backbone_snapshot)
        if self._decoder_snapshot is not None and self.aux_branch is not None:
            self.aux_branch.load_state_dict(self._decoder_snapshot, strict=False)

    # ---------------------------------------------------- adaptation
    def _tta_adapt(self, data_dict: Dict) -> float:
        """
        K SGD steps on (backbone + decoders) using L_aux.
        Returns the post-adaptation loss for logging.
        """
        loss_last = 0.0
        for _ in range(self.tta_n_steps):
            self._tta_optimizer.zero_grad(set_to_none=True)
            rec_tuple = self.aux_branch(data_dict)
            loss = self.rec_loss(rec_tuple, data_dict)
            loss.backward()
            self._tta_optimizer.step()
            loss_last = loss.item()
        return loss_last

    # ------------------------------------------------------------------ run
    def run(self):
        assert self.test_loader is not None
        assert self._tta_optimizer is not None, \
            'Call register_aux() after register_model().'

        self._load_checkpoints()
        self._save_snapshot()

        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        pbar = tqdm(enumerate(self.test_loader),
                    total=len(self.test_loader), ncols=180)

        for iteration, data_dict in pbar:
            torch.cuda.synchronize(); timer.add_prepare_time()
            self.iteration = iteration + 1
            data_dict = _prep(data_dict, self.cfg)
            self.before_test_step(self.iteration, data_dict)
            torch.cuda.synchronize(); timer.add_prepare_time()

            # 1. Adapt backbone + decoders for K steps on L_aux
            self._set_adapt_mode()
            torch.set_grad_enabled(True)
            tta_loss = self._tta_adapt(data_dict)
            torch.set_grad_enabled(False)
            self._set_eval_mode()

            # 2. Registration forward with adapted weights
            output_dict = self.test_step(self.iteration, data_dict)
            output_dict['tta_loss'] = float(tta_loss)
            torch.cuda.synchronize(); timer.add_process_time()

            # 3. Evaluate
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            self.logger.critical(
                self.summary_string(self.iteration, data_dict, output_dict, result_dict)
                + f', tta_loss={tta_loss:.4f}, {timer.tostring()}'
            )

            # 4. Episodic reset — next pair starts from meta-trained θ
            self._restore_snapshot()
            torch.cuda.empty_cache()

        self.after_test_epoch()
        self.logger.critical(
            get_log_string(result_dict=summary_board.summary(), timer=timer)
        )

    # ------------------------------------------------------------------
    # Computational stats variant — same logic, profiled
    # ------------------------------------------------------------------
    def run_computational_stats(self):
        assert self.test_loader is not None
        assert self._tta_optimizer is not None, \
            'Call register_aux() after register_model().'

        self._load_checkpoints()
        self._save_snapshot()

        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()

        inference_times = []
        flops_list = []
        max_flop_samples = 100
        peak_gpu_list = []

        pbar = tqdm(enumerate(self.test_loader),
                    total=len(self.test_loader), ncols=180)

        for iteration, data_dict in pbar:
            torch.cuda.synchronize(); timer.add_prepare_time()
            self.iteration = iteration + 1
            data_dict = _prep(data_dict, self.cfg)
            self.before_test_step(self.iteration, data_dict)
            torch.cuda.synchronize(); timer.add_prepare_time()

            profile_flops = iteration < max_flop_samples

            if profile_flops:
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)

                with profile(
                    activities=activities,
                    with_flops=True,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    self._set_adapt_mode()
                    torch.set_grad_enabled(True)
                    tta_loss = self._tta_adapt(data_dict)
                    torch.set_grad_enabled(False)
                    self._set_eval_mode()

                    output_dict = self.test_step(self.iteration, data_dict)
                    output_dict['tta_loss'] = float(tta_loss)
                    torch.cuda.synchronize()
                    t1 = time.time()

                total_flops = 0
                for evt in prof.key_averages():
                    if hasattr(evt, "flops") and evt.flops is not None:
                        total_flops += evt.flops
                flops_list.append(total_flops / 1e9)
            else:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                self._set_adapt_mode()
                torch.set_grad_enabled(True)
                tta_loss = self._tta_adapt(data_dict)
                torch.set_grad_enabled(False)
                self._set_eval_mode()

                output_dict = self.test_step(self.iteration, data_dict)
                output_dict['tta_loss'] = float(tta_loss)
                torch.cuda.synchronize()
                t1 = time.time()
                inference_times.append(t1 - t0)
                peak_gpu_list.append(torch.cuda.max_memory_allocated() / 1024**2)

            timer.add_process_time()

            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            self.logger.critical(
                self.summary_string(self.iteration, data_dict, output_dict, result_dict)
                + f', tta_loss={tta_loss:.4f}, {timer.tostring()}'
            )

            self._restore_snapshot()
            torch.cuda.empty_cache()

        self.after_test_epoch()
        self.logger.critical(
            get_log_string(result_dict=summary_board.summary(), timer=timer)
        )

        if len(inference_times) > 0:
            times = np.array(inference_times)
            self.logger.critical(
                f"Mean inference time per sample (no FLOP profiling): "
                f"{times.mean():.6f} s (std {times.std():.6f} s, n={len(times)})"
            )
        if len(peak_gpu_list) > 0:
            peak_gpu_arr = np.array(peak_gpu_list)
            self.logger.critical(
                f"Max GPU usage (no FLOP profiling): {peak_gpu_arr.max():.6f} MB"
            )
        if len(flops_list) > 0:
            flops_arr = np.array(flops_list)
            self.logger.critical(
                f"Mean GFLOPs over first {len(flops_arr)} samples: "
                f"{flops_arr.mean():.3f} G (std {flops_arr.std():.3f} G, "
                f"min {flops_arr.min():.3f} G, max {flops_arr.max():.3f} G)"
            )