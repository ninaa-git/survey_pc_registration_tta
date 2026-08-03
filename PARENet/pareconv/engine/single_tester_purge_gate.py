import pdb
from typing import Dict

import torch
import ipdb
from tqdm import tqdm

from pareconv.engine.base_tester import BaseTester
from pareconv.utils.summary_board import SummaryBoard
from pareconv.utils.timer import Timer
from pareconv.utils.common import get_log_string
from pareconv.utils.torch import release_cuda, to_cuda

import time
import numpy as np
from torch.profiler import profile, ProfilerActivity


class SingleTesterPurgeGate(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def _run_purge_iteration(self, iteration, data_dict, *, track_timer=False, timer=None):
        """Try purge sizes and keep the output with the best IR_final."""
        best_result_dict = None
        best_output_dict = None
        for purge_size in [0, 0.1, 0.5]:
            torch.cuda.synchronize()
            output_dict = self.test_step(iteration, data_dict, purge_size)
            torch.cuda.synchronize()
            if track_timer and timer is not None:
                timer.add_process_time()
            result_dict = self.eval_step(iteration, data_dict, output_dict)
            ir_final = result_dict['IR_final']
            if not np.isnan(ir_final) and (
                best_result_dict is None
                or np.isnan(best_result_dict['IR_final'])
                or ir_final > best_result_dict['IR_final']
            ):
                best_result_dict = result_dict
                best_output_dict = output_dict
                best_result_dict['purge_size'] = purge_size
            if np.isnan(ir_final):
                output_dict = self.test_step(iteration, data_dict, 0.0)
                torch.cuda.synchronize()
                if track_timer and timer is not None:
                    timer.add_process_time()
                result_dict = self.eval_step(iteration, data_dict, output_dict)
                best_result_dict = result_dict
                best_output_dict = output_dict
                best_result_dict['purge_size'] = 0.0
        return best_output_dict, best_result_dict
        
    def run_computational_stats(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
 
        # --- NEW: storage for stats ---
        inference_times = []   # pure inference times (no FLOP profiling)
        flops_list = []        # GFLOPs for profiled samples
        max_flop_samples = 100  # number of samples to use for FLOP stats (tune as you like)
        peak_gpu_list = []
 
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations, ncols=180)
        for iteration, data_dict in pbar:
            # on start
            torch.cuda.synchronize()
            timer.add_prepare_time()
 
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
 
            self.before_test_step(self.iteration, data_dict)
 
            # --------- test step: timing and/or FLOPs ----------
            profile_flops = iteration < max_flop_samples  # first N iterations -> FLOP profiling
 
            if profile_flops:
                # FLOP profiling (this adds overhead; we won't use its time for avg inference)
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
                    output_dict, result_dict = self._run_purge_iteration(
                        self.iteration, data_dict, track_timer=True, timer=timer,
                    )
                    torch.cuda.synchronize()
                    t1 = time.time()
 
                # compute total FLOPs for this iteration
                total_flops = 0
                for evt in prof.key_averages():
                    if hasattr(evt, "flops") and evt.flops is not None:
                        total_flops += evt.flops
 
                gflops = total_flops / 1e9
                flops_list.append(gflops)
                # we do NOT append (t1 - t0) to inference_times because profiler slows it down
            else:
                # pure inference timing (no FLOP profiling)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                output_dict, result_dict = self._run_purge_iteration(
                    self.iteration, data_dict, track_timer=True, timer=timer,
                )
                torch.cuda.synchronize()
                t1 = time.time()
                inference_times.append(t1 - t0)
                peak_gpu = torch.cuda.max_memory_allocated() / 1024**2
                peak_gpu_list.append(peak_gpu)
 
            torch.cuda.synchronize()
            timer.add_process_time()
            # --------------------------------------------------
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            # pbar.set_description(message)
            self.logger.critical(message)
            torch.cuda.empty_cache()
 
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
 
        # --- NEW: final time & FLOP stats ---
        if len(inference_times) > 0:
            times = np.array(inference_times)
            mean_t, std_t = times.mean(), times.std()
            self.logger.critical(
                f"Mean inference time per sample (no FLOP profiling): "
                f"{mean_t:.6f} s (std {std_t:.6f} s, n={len(times)})"
            )
 
        if len(peak_gpu_list) > 0:
            peak_gpu_arr = np.array(peak_gpu_list)
            max_p = peak_gpu_arr.max()
            self.logger.critical(
                f"Max GPU usage (no FLOP profiling): "
                f"{max_p:.6f} MB)"
            )
 
        if len(flops_list) > 0:
            flops_arr = np.array(flops_list)
            mean_g, std_g = flops_arr.mean(), flops_arr.std()
            self.logger.critical(
                f"Mean GFLOPs over first {len(flops_arr)} samples: "
                f"{mean_g:.3f} G (std {std_g:.3f} G, "
                f"min {flops_arr.min():.3f} G, max {flops_arr.max():.3f} G)"
            )
   

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations, ncols=180)
        for iteration, data_dict in pbar:
            # on start
            torch.cuda.synchronize()
            timer.add_prepare_time()
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            best_result_dict = None
            best_output_dict = None
            for purge_size in [0, 0.1, 0.5]:
                torch.cuda.synchronize()
                output_dict = self.test_step(self.iteration, data_dict, purge_size)
                torch.cuda.synchronize()
                timer.add_process_time()
                # eval step
                result_dict = self.eval_step(self.iteration, data_dict, output_dict)
                ir_final = result_dict['IR_final']
                if not np.isnan(ir_final) and (
                    best_result_dict is None
                    or np.isnan(best_result_dict['IR_final'])
                    or ir_final > best_result_dict['IR_final']
                ):
                    best_result_dict = result_dict
                    best_output_dict = output_dict
                    best_result_dict['purge_size'] = purge_size
                if np.isnan(ir_final):
                    output_dict = self.test_step(self.iteration, data_dict, 0.0)
                    torch.cuda.synchronize()
                    timer.add_process_time()
                    # eval step
                    result_dict = self.eval_step(self.iteration, data_dict, output_dict)
                    best_result_dict = result_dict
                    best_output_dict = output_dict
                    best_result_dict['purge_size'] = 0.0

            result_dict = best_result_dict
            output_dict = best_output_dict
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            # pbar.set_description(message)
            self.logger.critical(message)
            torch.cuda.empty_cache()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
    