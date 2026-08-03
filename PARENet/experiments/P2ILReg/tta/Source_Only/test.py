import argparse
import os.path as osp
import pdb
import time
import os
import json

import torch
import numpy as np
import pickle
from pareconv.engine import SingleTester
from pareconv.utils.torch import release_cuda
from pareconv.utils.common import ensure_dir, get_log_string

from config import make_cfg
from model import create_model
from loss import Evaluator

from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)
from common_utils.parser import make_parser
from common_utils.visualisation import visualise_registration_SelfP2IR
from dataset import test_data_loader, corrupted_test_data_loader


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
        model = create_model(cfg).cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.register_model(model)
        # evaluator
        self.evaluator = Evaluator(cfg).cuda()
        
        # preparation
        self.output_dir = osp.join(cfg.registration_dir)
        ensure_dir(self.output_dir)
        if self.args.viz_iter:
            self.viz_dir = osp.join(cfg.viz_dir)
            ensure_dir(self.viz_dir)

        if cfg.data.dataset == 'real':
            out_file = osp.join(self.output_dir, f'test_{cfg.data.dataset}.json')
        else:
            out_file = osp.join(self.output_dir, f'test_{self.args.corruption}_{self.args.severity}.json')
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
            print(f'visualised iteration: {iteration}')



def main():

    
    cfg = make_cfg()
    torch.set_float32_matmul_precision('high')
    tester = Tester(cfg)
    #tester.run_computational_stats()
    tester.run()


if __name__ == '__main__':
    main()