import pdb
import sys
import argparse
import os.path as osp
import time
import json
import abc
import torch
import ipdb

from pareconv.utils.torch import clean_checkpoint_state_dict, initialize
from pareconv.engine.logger import Logger

def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    return parser

class BaseTester(abc.ABC):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        # parser
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()
        # logger
        log_file = osp.join(cfg.log_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file)
        self.cfg = cfg
        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)

        # find snapshot
        if self.args.snapshot is None:
            raise RuntimeError('Snapshot is not specified.')

        # print config
        message = 'Configs:\n' + json.dumps(cfg, indent=4)
        self.logger.info(message)

        # cuda and distributed
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.cudnn_deterministic = cudnn_deterministic
        self.seed = cfg.seed
        initialize(seed=self.seed, cudnn_deterministic=self.cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        self.test_loader = None
        self.saved_states = {}

    """ def load_snapshot(self, snapshot, strict=True):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        cleaned = clean_checkpoint_state_dict(state_dict)

        # Load into the unwrapped model -> works whether self.model is DDP-wrapped or not.
        target = self.model.module if hasattr(self.model, 'module') else self.model

        snapshot_keys = set(cleaned.keys())
        model_keys = set(target.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys

        if missing_keys:
            self.logger.warning(f'Missing keys: {missing_keys}')
        if unexpected_keys:
            self.logger.warning(f'Unexpected keys: {unexpected_keys}')

        target.load_state_dict(cleaned, strict=strict)
        self.logger.info('Model has been loaded.') """
    def load_snapshot(self, snapshot, strict=True):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))

        # Checkpoints may nest the weights under a key, or be the raw state dict.
        if isinstance(state_dict, dict):
            for k in ('model', 'state_dict'):
                if k in state_dict and isinstance(state_dict[k], dict):
                    state_dict = state_dict[k]
                    break

        cleaned = clean_checkpoint_state_dict(state_dict)

        target = self.model.module if hasattr(self.model, 'module') else self.model
        model_keys = set(target.state_dict().keys())

        def strip_module(sd):
            return {k[len('module.'):] if k.startswith('module.') else k: v
                    for k, v in sd.items()}

        def add_module(sd):
            return {('module.' + k): v for k, v in sd.items()}

        # Try, in order: as-is, module-stripped, module-added. Pick the variant
        # whose keys best overlap what the model expects.
        candidates = [cleaned, strip_module(cleaned), add_module(cleaned)]
        best = max(candidates, key=lambda sd: len(model_keys & set(sd.keys())))

        snapshot_keys = set(best.keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if missing_keys:
            self.logger.warning(f'Missing keys: {missing_keys}')
        if unexpected_keys:
            self.logger.warning(f'Unexpected keys: {unexpected_keys}')

        target.load_state_dict(best, strict=strict)
        self.logger.info('Model has been loaded.')

    def register_model(self, model):
        r"""Register model. DDP is automatically used."""
        self.model = model
        message = 'Model description:\n' + str(model)
        self.logger.info(message)
        return model

    def register_loader(self, test_loader):
        r"""Register data loader."""
        self.test_loader = test_loader

    @abc.abstractmethod
    def run(self):
        raise NotImplemented
