import os
import glob
import tqdm
import copy
import imageio
import random
import tensorboardX
import numpy as np

import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from .base import Attack


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def _get_param(nerf):
    params = []
    for name, param in nerf.named_parameters():
        if name in ('sdf', 'deform'): continue
        params.append(param.view(-1))
    return torch.concat(params)

def _set_param(nerf, w):
    index = 0
    for name, param in nerf.named_parameters():
        if name in ('sdf', 'deform'): continue
        if len(param) > 0:
            param.data = w[index:index+param.view(-1).shape[0]].data.view(param.shape)
            index = index + param.view(-1).shape[0]
        

_img2mse = lambda x, y : torch.mean((x - y) ** 2)
_mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

class Pruning(Attack):
    def __init__(self, 
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ori_model=None,
                 **kwargs):
        
        super().__init__(
                 argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 guidance, # guidance network
                 criterion, # loss function, if None, assume inline implementation in train_step
                 optimizer, # optimizer
                 ema_decay, # if use EMA, set the decay
                 lr_scheduler, # scheduler
                 metrics, # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank, # which GPU am I
                 world_size, # total num of GPUs
                 device, # device to use, usually setting to None is OK. (auto choose device)
                 mute, # whether to mute all print
                 fp16, # amp optimize level
                 eval_interval, # eval once every $ epoch
                 max_keep_ckpt, # max num of saved ckpts in disk
                 workspace, # workspace to save logs & ckpts
                 best_mode, # the smaller/larger result, the better
                 use_loss_as_metric, # use loss as the first metric
                 report_metric_at_train, # also report metrics at training
                 use_checkpoint, # which ckpt to use at init time
                 use_tensorboardX, # whether to use tensorboard for logging
                 scheduler_update_every_step, # whether to call scheduler.step() after every train step
                 ori_model,
                 **kwargs)

        ori_model.load_state_dict(self.model.state_dict(), strict=False)
        ori_model.tet_scale = self.model.tet_scale
        ori_model.verts *= self.model.tet_scale
        if opt.cuda_ray:
            ori_model.mean_density = self.model.mean_density
        ori_model.set_idx()
        self.ori_model = ori_model
        
        for name, param in self.model.named_parameters():
            p2 = self.ori_model.get_parameter(name)
            assert (p2 - param).abs().max() < 1e-4
    
    def train(self, train_loader, valid_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        percentage = 12
        
        w = _get_param(self.model).detach().cpu().numpy()
        threshold = np.percentile(np.abs(w), percentage)
        idx = np.abs(w) < threshold
        w[idx] = 0
        
        _set_param(self.model, torch.tensor(w, device=self.device))
        acc, rgb = self.watermark.eval_step(self.model)
        # print(rgb[0].shape)
        # exit()
        torchvision.utils.save_image(rgb[0], "1.png", normalize=True, range=(0,1))
        _, rgb_target = self.watermark.eval_step(self.ori_model)

        img_mse = _img2mse(rgb[0], rgb_target[0])
        img_PSNR = _mse2psnr(img_mse)
        self.log(f"percentage={percentage}\nacc={acc[0]}\nPSNR={img_PSNR}")

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()