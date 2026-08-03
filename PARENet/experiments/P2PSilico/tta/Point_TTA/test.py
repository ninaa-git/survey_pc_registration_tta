import argparse
import os
import os.path as osp
import json

import numpy as np
from easydict import EasyDict

from config import make_cfg
from tta.Source_Only.model import create_model
from loss import Evaluator
from tta.Point_TTA.rec_aux import RecAux

from pareconv.meta_engine.single_tester import SingleTester, SingleTesterTTA

from pareconv.utils.common import ensure_dir, get_log_string
from dataset import corrupted_test_data_loader

from common_utils.parser import make_parser
from common_utils.visualisation import visualise_registration

# ---------------------------------------------------------------------------
# Shared hooks
# ---------------------------------------------------------------------------

class _RegistrationHooks:

    def _setup_output(self, args, cfg):
        self.output_dir = cfg.registration_dir
        ensure_dir(self.output_dir)
        if self.args.viz_iter:
            self.viz_dir = osp.join(cfg.viz_dir)
            ensure_dir(self.viz_dir)
        suffix = args.corruption if args.severity is None \
            else f'{args.corruption}_{args.severity}'
        self.out_file = osp.join(
            self.output_dir, f'test_{args.mode}_{suffix}_{args.lr}.json'
        )
        if osp.exists(self.out_file):
            self.logger.warning(f'Overwriting: {self.out_file}')
            #os.remove(self.out_file)

    def test_step(self, iteration, data_dict):
        return self.model(data_dict)

    def eval_step(self, iteration, data_dict, output_dict):
        return self.evaluator(output_dict, data_dict)

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene = data_dict['scene_name']
        ref   = data_dict['ref_frame']
        src   = data_dict['src_frame']
        return (f'{scene}, id0: {ref}, id1: {src}, '
                + get_log_string(result_dict=result_dict))

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        entry = {
            'iteration': iteration,
            'PIR':       float(result_dict['PIR']),
            'IR':        float(result_dict['IR']),
            'RRE':       float(result_dict['RRE']),
            'RTE':       float(result_dict['RTE']),
            'RMSE':      float(result_dict['RMSE']),
            'RE_markers': (
                float(result_dict['RE_markers'])
                if np.isscalar(result_dict['RE_markers'])
                else result_dict['RE_markers'].tolist()
            ),
        }
        with open(self.out_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        if iteration in self.args.viz_iter:
            visualise_registration(output_dict, data_dict, iteration, self.viz_dir, self.args.corruption, self.args.severity)


# ---------------------------------------------------------------------------
# Tester: standard
# ---------------------------------------------------------------------------

class TesterStandard(_RegistrationHooks, SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        cfg.test.batch_size = 1
        
        loader, neighbor_limits = corrupted_test_data_loader(self.args, cfg)
        self.register_loader(loader)
        model = create_model(cfg).cuda()
        self.register_model(model)
        self.evaluator = Evaluator(cfg).cuda()
        self._setup_output(self.args, cfg)
        print(f'Parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M')


# ---------------------------------------------------------------------------
# Tester: TTA
# ---------------------------------------------------------------------------

class TesterTTA(_RegistrationHooks, SingleTesterTTA):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        cfg.test.batch_size = 1

        if self.args.meta_snapshot is None:
            raise ValueError(
                '--meta_snapshot is required for --mode pointtta.\n'
                'Provide the best_model.pth.tar from the meta phase of trainval.py.'
            )

        # Push CLI args into cfg.tta so SingleTesterTTA can read them.
        if not hasattr(cfg, 'tta'):
            cfg.tta = EasyDict()
        cfg.tta.lr    = self.args.lr
        cfg.tta.niter = self.args.niter

        loader, neighbor_limits = corrupted_test_data_loader(self.args, cfg)
        self.register_loader(loader)

        model = create_model(cfg).cuda()
        self.register_model(model)

        # RecAux shares its .backbone with model.backbone (same Python object).
        aux_branch = RecAux(model, cfg).cuda()
        self.register_aux(aux_branch)

        self.evaluator = Evaluator(cfg).cuda()
        self._setup_output(self.args, cfg)
        print(f'Parameters: {sum(p.nelement() for p in model.parameters())/1e6:.2f}M')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import torch
    #torch.backends.cuda.preferred_linalg_library('magma')
    args = make_parser().parse_args()
    cfg = make_cfg()

    tester = {
        'standard': TesterStandard,
        'pointtta': TesterTTA,
    }[args.mode](cfg)

    #tester.run()
    tester.run_computational_stats()


if __name__ == '__main__':
    main()