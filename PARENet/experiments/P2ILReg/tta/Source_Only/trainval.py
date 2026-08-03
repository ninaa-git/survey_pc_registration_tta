import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "LOCAL_RANK" in os.environ:
    import sys
    if not any(arg.startswith("--local_rank") for arg in sys.argv):
        sys.argv.append(f"--local_rank={os.environ['LOCAL_RANK']}")
import torch._inductor.config
import pdb
import time
import torch
import torch.optim as optim
from pareconv.engine import EpochBasedTrainer
from config import make_cfg
from dataset import train_valid_data_loader
from tta.Source_Only.model import create_model
from loss import OverallLoss, Evaluator
from pareconv.utils.common import print_model_parameters
from common_utils.parser import make_hyperparameters_training_parser
from common_utils.parser import make_launcher_parser

import wandb.util
wandb.util.working_set = lambda: iter([])


#parser=make_hyperparameters_training_parser("Training Registration Pipeline"),

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_launcher_parser("Training Registration Pipeline"), max_epoch=cfg.optim.max_epoch, cudnn_deterministic=False, run_grad_check=False, autograd_anomaly_detection=False)
        
        cfg.data.dataset = 'syn'
        torch.set_float32_matmul_precision('high')
        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed) #self.distributed is False

        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()

        model = self.register_model(model)

        if self.distributed:
            model.module.backbone    = torch.compile(model.module.backbone,    mode="default", dynamic=True)
            model.module.transformer = torch.compile(model.module.transformer,    mode="default", dynamic=True)
        else:
            model.backbone    = torch.compile(model.backbone,    mode="default", dynamic=True)
            model.transformer = torch.compile(model.transformer,    mode="default", dynamic=True)
        
        if self.local_rank == 0:
            print_model_parameters(model)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, fused=True)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

        self.amp_dtype = torch.bfloat16
        self.use_amp = True 
        
    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        with torch.no_grad():
            output_dict = self.model(data_dict)
            loss_dict = self.loss_func(output_dict, data_dict)
            result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()



if __name__ == '__main__':
    main()
