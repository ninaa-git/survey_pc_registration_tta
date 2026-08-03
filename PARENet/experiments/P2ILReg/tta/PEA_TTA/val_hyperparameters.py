import argparse
import os.path as osp
import pdb
import time
import os
import json

import numpy as np
import pickle
import torch
from pareconv.engine import SingleTester
from pareconv.utils.torch import release_cuda
from pareconv.utils.common import ensure_dir, get_log_string

#from dataset import test_data_loader
from config import make_cfg
from tta.PEA_TTA.model import create_model
from loss import Evaluator

from dataset import train_valid_data_loader
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)

from common_utils.parser import make_parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        cfg.test.batch_size = 1

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, False)
        neighbor_limits = cfg.backbone.num_neighbors
        
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(val_loader)

        # model
        model = create_model(cfg).cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.register_model(model)

        # ── WCT coarse-feature alignment (PEA eq. 3-4) ───────────────────────
        # Activated when --align_weight > 0.  Loads precomputed source and
        # target statistics from generate_stats.py and installs them on
        # the model so every forward pass applies the alignment.
        if self.args.align_weight > 0.0:
            stats_file = osp.join(self.args.align_stats_dir, 'source_feats_syn.pth')

            if not osp.exists(stats_file):
                self.logger.warning(
                    f'Alignment stats not found: {stats_file}. '
                    f'Generating them now (this may take a while)...'
                )
                try:
                    from generate_stats import runner as generate_pea_stats
                    # generate_stats expects --checkpoint; our pipeline uses --snapshot
                    if not hasattr(self.args, 'checkpoint') or getattr(self.args, 'checkpoint') is None:
                        setattr(self.args, 'checkpoint', getattr(self.args, 'snapshot', None))
                    generate_pea_stats(self.args, cfg)
                except Exception as e:
                    raise FileNotFoundError(
                        f'Alignment stats not found: {stats_file}\n'
                        f'Auto-generation failed. You can run generate_stats.py manually.\n'
                        f'Original error: {type(e).__name__}: {e}'
                    ) from e

                if not osp.exists(stats_file):
                    raise FileNotFoundError(
                        f'Alignment stats generation finished but the expected file was not created: {stats_file}\n'
                        f'Check that --align_stats_dir matches the output directory.'
                    )

            self.logger.info(f'Loading alignment stats from: {stats_file}')
            pea_results = torch.load(stats_file, weights_only=False)

            # 'ref_feats_c' is the coarse RI head output — the feature that
            # enters the transformer, so it's the right one to align.
            stage_stats = pea_results['ref_feats_c']
            # Backward-compat: older caches stored raw stats dict directly
            # (i.e. {'mean','var','cov',...}) instead of {'src': {...}}.
            model.set_alignment_stats(
                src_stats=stage_stats['src'],
                weight=self.args.align_weight,
                momentum=self.args.align_momentum,
            )
            self.logger.info(
                f'WCT alignment active  |  stage=ref_feats_c  '
                f'w={self.args.align_weight}  '
                f'm={self.args.align_momentum}  '
            )
        else:
            self.logger.info('WCT alignment disabled (--align_weight 0)')
        # ─────────────────────────────────────────────────────────────────────


        self.evaluator = Evaluator(cfg).cuda()
        
        # preparation
        self.output_dir = osp.join(cfg.ft_validation_dir)
        ensure_dir(self.output_dir)
        if self.args.corruption == 'clean':
            out_file = osp.join(self.output_dir, f'val_{self.args.corruption}.json')
        else:
            out_file = osp.join(self.output_dir, f'val_{self.args.corruption}_{self.args.severity}.json')

        if osp.exists(out_file):
            self.logger.warning(
                f"Output file already exists and will be overwritten: {out_file}"
            )
            os.remove(out_file)

        self.out_file = out_file

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
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
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        output_dir = osp.join(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        corruption = self.args.corruption
        severity = self.args.severity
        momentum = self.args.align_momentum
        
        out_file = osp.join(output_dir, f'val_{momentum}.json')


        PIR = result_dict['PIR']
        IR = result_dict['IR']
        RRE = result_dict['RRE']
        RTE = result_dict['RTE']
        RMSE = result_dict['RMSE']

        entry = {
            'iteration': iteration,
            'PIR': float(PIR),
            'IR': float(IR),
            'RRE': float(RRE),
            'RTE': float(RTE),
            'RMSE': float(RMSE),
        }

        with open(out_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def main():

    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()