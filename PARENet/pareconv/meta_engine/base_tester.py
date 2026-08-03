import sys
import argparse
import os.path as osp
import time
import json
import abc
import torch

from pareconv.utils.torch import initialize
from pareconv.meta_engine.logger import Logger


def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    existing = {a.dest for a in parser._actions}

    if 'snapshot' not in existing:
        parser.add_argument('--snapshot', default='../../pretrain/3dmatch.pth.tar')
    if 'meta_snapshot' not in existing:
        parser.add_argument('--meta_snapshot', default=None)

    return parser


class BaseTester(abc.ABC):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()

        log_file = osp.join(cfg.log_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file)
        self.cfg = cfg

        self.logger.info('Command executed: ' + ' '.join(sys.argv))

        if self.args.snapshot is None and getattr(self.args, 'meta_snapshot', None) is None:
            raise RuntimeError('Either --snapshot or --meta_snapshot must be specified.')

        try:
            self.logger.info('Configs:\n' + json.dumps(cfg, indent=4))
        except TypeError:
            self.logger.info('Configs:\n' + str(cfg))

        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.cudnn_deterministic = cudnn_deterministic
        self.seed = cfg.seed
        initialize(seed=self.seed, cudnn_deterministic=self.cudnn_deterministic)

        self.model = None
        self.iteration = None
        self.test_loader = None
        self.saved_states = {}

    def load_snapshot(self, snapshot):
        """Load model backbone + heads from a standard pretrained checkpoint."""
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'), weights_only=False)
        assert 'model' in state_dict, 'No model key found in checkpoint.'
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.logger.info('Model loaded.')

    def load_meta_snapshot(self, snapshot, aux_branch=None):
        '''
        Load from a meta_trainval.py checkpoint (best_model.pth.tar).
        '''
        self.logger.info(f'Loading meta checkpoint from "{snapshot}".')
        ckpt = torch.load(snapshot, map_location=torch.device('cpu'), weights_only=False)

        assert 'model' in ckpt, 'Meta checkpoint has no "model" key.'
        model_state = {
            k.replace('_orig_mod.', ''): v
            for k, v in ckpt['model'].items()
        }
        self.model.load_state_dict(model_state, strict=True)
        self.logger.info('Backbone + heads loaded.')

        if aux_branch is not None:
            if 'rec_branch' in ckpt:
                aux_state = {
                    k: v for k, v in ckpt['rec_branch'].items()
                    if not k.startswith('backbone.')
                }
                aux_branch.load_state_dict(aux_state, strict=False)
                self.logger.info('Reconstruction decoders loaded.')
            else:
                self.logger.warning('No rec_branch in checkpoint — decoders keep random init.')

        train_phase = ckpt.get('train_phase', 'unknown')
        self.logger.info(f'(checkpoint train_phase: {train_phase})')

    def register_model(self, model):
        self.model = model
        self.logger.info('Model description:\n' + str(model))
        return model

    def register_loader(self, test_loader):
        self.test_loader = test_loader

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError