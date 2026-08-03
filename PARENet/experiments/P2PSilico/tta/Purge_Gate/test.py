import os.path as osp
import time
import os
import json

import numpy as np
import torch
from pareconv.engine import SingleTesterPurgeGate
from pareconv.utils.common import ensure_dir, get_log_string

from config import make_cfg
from tta.Purge_Gate.model import create_model
from loss import Evaluator


from dataset import corrupted_test_data_loader
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)
from common_utils.parser import make_parser
from generate_stats import load_or_collect
from common_utils.visualisation import visualise_registration


class Tester(SingleTesterPurgeGate):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        cfg.test.batch_size = 1

        # dataloader
        start_time = time.time()
        tta_loader, neighbor_limits = corrupted_test_data_loader(self.args, cfg)
        neighbor_limits = cfg.backbone.num_neighbors

        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(tta_loader)

        # model
        model = create_model(cfg).cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        self.register_model(model)

        # load source intermediate statistics for purge gating
        cache_path = (
            f"intermediate_features/purge_gate_stats/"
            f"{self.args.stats_mode}_{self.args.feature}_intermediates.pth"
        )
        results = load_or_collect(
            model=model,
            cfg=cfg,
            cache_path=cache_path,
            feature_keys=[self.args.feature],
            mode=self.args.stats_mode,
            force_regenerate=False,
        )
        self.source_intermediates_stats = results[self.args.feature]
        self.logger.info(
            f'Loaded source stats  |  feature={self.args.feature}  '
            f'mode={self.args.stats_mode}'
        )

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        self.output_dir = osp.join(cfg.registration_dir)
        ensure_dir(self.output_dir)

        if self.args.corruption == 'clean':
            out_file = osp.join(
                self.output_dir,
                f'test_{self.args.corruption}'
                f'.json',
            )
        else:
            out_file = osp.join(
                self.output_dir,
                f'test_{self.args.corruption}'
                f'_{self.args.severity}.json',
            )

        if osp.exists(out_file):
            self.logger.warning(
                f"Output file already exists and will be overwritten: {out_file}"
            )
            os.remove(out_file)

        self.out_file = out_file

        if self.args.viz_iter:
            self.viz_dir = osp.join(cfg.viz_dir)
            ensure_dir(self.viz_dir)

    def test_step(self, iteration, data_dict, purge_size):
        output_dict = self.model.forward_prototype_purge(
            data_dict,
            self.args.feature,
            self.source_intermediates_stats,
            purge_size,
        )
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        RRE = result_dict['RRE']
        RTE = result_dict['RTE']
        RMSE = result_dict['RMSE']
        RE_markers = result_dict['RE_markers']
        purge_size = result_dict['purge_size']
        IR_final = result_dict['IR_final']

        entry = {
            'iteration': iteration,
            'RRE': float(RRE),
            'RTE': float(RTE),
            'RMSE': float(RMSE),
            'purge_size': float(purge_size),
            'IR_final': float(IR_final),
            'RE_markers': (
                float(RE_markers) if np.isscalar(RE_markers) else RE_markers.tolist()
            ),
            
            
        }

        with open(self.out_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        if iteration in self.args.viz_iter:
            visualise_registration(output_dict, data_dict, iteration, self.viz_dir, self.args.corruption, self.args.severity)


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run_computational_stats()
    #tester.run()


if __name__ == '__main__':
    main()