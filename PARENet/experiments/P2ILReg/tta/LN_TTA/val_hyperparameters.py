import argparse
import os.path as osp
import pdb
import time
import os
import json

import numpy as np
import pickle
from pareconv.engine import SingleTester
from pareconv.utils.torch import release_cuda
from pareconv.utils.common import ensure_dir, get_log_string

import torch
from config import make_cfg
from model import create_model
from loss import Evaluator

from dataset import train_valid_data_loader
from ln_tta import LN_TTA_Manager
from generate_stats import collect_source_ln_stats

from pareconv.modules.transformer.rpe_transformer_LN import TTAContext

from common_utils.parser import make_parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        cfg.test.batch_size = 1
        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, False)
        
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(val_loader)

        # model
        ctx = TTAContext()
        model = create_model(cfg).cuda()
        # inject ctx into the RPEConditionalTransformer and all its RPETransformerLayers
        model.transformer.transformer.ctx = ctx
        for layer in model.transformer.transformer.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'norm'):
                if hasattr(layer.attention.norm, 'ctx'):
                    layer.attention.norm.ctx = ctx
                if hasattr(layer.output, 'norm') and hasattr(layer.output.norm, 'ctx'):
                    layer.output.norm.ctx = ctx
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.register_model(model)
        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # Source stats (offline, collected once from clean training data)
        src_stats_path = osp.join(
            'intermediate_features', 'ln_stats', 'ln_source_stats.pth'
        )
        source_stats = collect_source_ln_stats(
            model, cfg, self.args, src_stats_path, force_regenerate=False
        )

        # Manager — target stats are collected online via EMA during test_step
        self.manager = LN_TTA_Manager(
            transformer     = model.transformer,
            source_stats    = source_stats,
            ctx             = ctx,
            mode            = 'oracle',
            layers_to_adapt = self.args.layers_to_adapt,
            momentum        = self.args.ln_momentum,
        )
        
        # preparation
        self.output_dir = osp.join(cfg.ft_validation_dir)
        ensure_dir(self.output_dir)
        out_file = osp.join(self.output_dir, f'val_{self.args.ln_momentum}.json')

        if osp.exists(out_file):
            self.logger.warning(
                f"Output file already exists and will be overwritten: {out_file}"
            )
            os.remove(out_file)

        self.out_file = out_file

    def test_step(self, iteration, data_dict):
        with torch.no_grad():
            output_dict = self.manager.adapt(self.model, data_dict)
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

        out_file = self.out_file

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