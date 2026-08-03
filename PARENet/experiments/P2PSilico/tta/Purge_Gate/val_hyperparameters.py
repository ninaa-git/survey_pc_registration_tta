import os.path as osp
import time
import os
import json

import numpy as np
import torch
from pareconv.engine import SingleTester
from pareconv.utils.common import ensure_dir, get_log_string

from config import make_cfg
from tta.Purge_Gate.model import create_model
from loss import Evaluator


from dataset import train_valid_data_loader
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)
from common_utils.parser import make_parser
from generate_stats import load_or_collect


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

        self.output_dir = osp.join(cfg.ft_validation_dir)
        ensure_dir(self.output_dir)

        out_file = osp.join(
            self.output_dir,
            f'purge_{self.args.purge_size}_{self.args.stats_mode}.json',
        )

        if osp.exists(out_file):
            self.logger.warning(
                f"Output file already exists and will be overwritten: {out_file}"
            )
            os.remove(out_file)

        self.out_file = out_file

    def test_step(self, iteration, data_dict):
        output_dict = self.model.forward_prototype_purge(
            data_dict,
            self.args.feature,
            self.source_intermediates_stats,
            self.args.purge_size,
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

        entry = {
            'iteration': iteration,
            'RRE': float(RRE),
            'RTE': float(RTE),
            'RMSE': float(RMSE),
        }

        with open(self.out_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()