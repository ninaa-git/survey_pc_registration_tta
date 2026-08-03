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

from dataset import test_data_loader, corrupted_test_data_loader
from ln_tta import LN_TTA_Manager
from generate_stats import collect_source_ln_stats

from pareconv.modules.transformer.rpe_transformer_LN import TTAContext

from common_utils.parser import make_parser
from common_utils.visualisation import visualise_registration_SelfP2IR


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        cfg.test.batch_size = 1

        # dataloader
        start_time = time.time()
        if cfg.data.dataset == 'real':
            tta_loader, neighbor_limits = test_data_loader(cfg, cfg.data.dataset)
        else:
            tta_loader, neighbor_limits = corrupted_test_data_loader(self.args, cfg)
        
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(tta_loader)

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
        self.output_dir = osp.join(cfg.registration_dir)
        ensure_dir(self.output_dir)
        if self.args.viz_iter:
            self.viz_dir = osp.join(cfg.viz_dir)
            ensure_dir(self.viz_dir)
            
        if cfg.data.dataset == 'real':
            out_file = osp.join(self.output_dir, f'test_{cfg.data.dataset}_{self.args.ln_momentum}.json')
        else:
            out_file = osp.join(self.output_dir, f'test_{self.args.corruption}_{self.args.severity}_{self.args.ln_momentum}.json')

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
        return get_log_string(result_dict=result_dict)

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        # The real-set evaluator returns {'CD', 'DICE'}; the syn-set evaluator
        # returns {'PIR','IR','RRE','RTE','RMSE',...}. Branch on what's present
        # so we never KeyError, and record every scalar that came back.
        entry = {
            'iteration': iteration,
            'BS': int(output_dict.get('batch_size', data_dict.get('batch_size', 1))),
        }
        for k, v in result_dict.items():
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                pass

        with open(self.out_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # NOTE: `visualise_registration` originally received `corruption` and
        # `severity`, which are not defined anywhere in this tester (leftover from
        # a corruption-robustness benchmark). They are removed here; adjust the
        # call to match your actual `visualise_registration` signature.
        if iteration in self.args.viz_iter:
            visualise_registration_SelfP2IR(output_dict, data_dict, iteration, self.viz_dir)



def main():
    cfg = make_cfg()
    torch.set_float32_matmul_precision('high')
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()