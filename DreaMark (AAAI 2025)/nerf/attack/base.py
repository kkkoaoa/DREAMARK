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

from ..watermark import ResidueMark, HiddenMark

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

class Attack(object):
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
        
        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
    
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:
            
            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()
        
        else:
            self.text_z = None
        
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)


        if not opt.lora:
            if opt.q_cond:
                from conditional_unet import CondUNet2DModel
                unetname = CondUNet2DModel
            else:
                from diffusers import UNet2DModel
                unetname = UNet2DModel
            self.unet = unetname(
                sample_size=64, # render height for NeRF in training, assert opt.h==opt.w==64
                in_channels=4,
                out_channels=4,
                layers_per_block=2,
                block_out_channels=(128, 256, 384, 512),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
            )    
        else:
            # use lora
            from lora_unet import UNet2DConditionModel     
            from diffusers.loaders import AttnProcsLayers
            from diffusers.models.attention_processor import LoRAAttnProcessor
            import einops
            if not opt.v_pred:
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet", local_files_only=True, low_cpu_mem_usage=False, device_map=None).to(device)
            else:
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", local_files_only=True, low_cpu_mem_usage=False, device_map=None).to(device)
            _unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in _unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = _unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = _unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            _unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(_unet.attn_processors)

            text_input = self.guidance.tokenizer(opt.text, padding='max_length', max_length=self.guidance.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            with torch.no_grad():
                text_embeddings = self.guidance.text_encoder(text_input.input_ids.to(self.guidance.device))[0]
            
            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                    self.text_embeddings = text_embeddings
                def forward(self,x,t,c=None,shading="albedo"):
                    textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                    return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
            self._unet = _unet
            self.lora_layers = lora_layers
            self.unet = LoraUnet().to(device)                     

        self.unet = self.unet.to(self.device)
        if not opt.lora:
            self.unet_optimizer = optim.Adam(self.unet.parameters(), lr=self.opt.unet_lr) # naive adam
        else:
            params = [
                {'params': self.lora_layers.parameters()},
                {'params': self._unet.camera_emb.parameters()},
                {'params': self._unet.lambertian_emb},
                {'params': self._unet.textureless_emb},
                {'params': self._unet.normal_emb},
            ] 
            self.unet_optimizer = optim.AdamW(params, lr=self.opt.unet_lr) # naive adam
        warm_up_lr_unet = lambda iter: iter / (self.opt.warm_iters*self.opt.K+1) if iter <= (self.opt.warm_iters*self.opt.K+1) else 1
        self.unet_scheduler = optim.lr_scheduler.LambdaLR(self.unet_optimizer, warm_up_lr_unet)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        # self.test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)


        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Opt: {opt}')
        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        self.buffer_imgs = None
        self.buffer_poses = None
        # watermark
        self.watermark = HiddenMark(opt, device=device, H=opt.h, W=opt.w)
        # self.watermark = ResidueMark(opt=opt, device=device, H=opt.h, W=opt.w, guidance=self.guidance)
        if opt.init_ckpt != '':
            checkpoint_dict = torch.load(opt.init_ckpt, map_location=device)
            self.watermark.load_state_dict(checkpoint_dict)


    def init_evalpose(self, loader):
        poses = None
        for data in loader:
            pose = data['pose']
            pose = pose.view(pose.shape[0],16).contiguous()
            if poses is None:
                poses = pose
            else:
                poses = torch.cat([poses, pose], dim = 0)
        return poses

    def add_buffer(self, latent, pose):
        pose = pose.view(pose.shape[0],16).contiguous()
        if self.buffer_imgs is None:
            self.buffer_imgs = latent
            self.buffer_poses = pose
        elif self.buffer_imgs.shape[0] < self.opt.buffer_size:
            self.buffer_imgs = torch.cat([self.buffer_imgs, latent], dim = 0)
            self.buffer_poses = torch.cat([self.buffer_poses, pose], dim = 0)
        else:
            self.buffer_imgs = torch.cat([self.buffer_imgs[1:], latent], dim = 0)
            self.buffer_poses = torch.cat([self.buffer_poses[1:], pose], dim = 0)            

    def sample_buffer(self, bs):
        idxs = torch.tensor(random.sample(range(self.opt.buffer_size), bs), device = self.device).long()
        s_imgs = torch.index_select(self.buffer_imgs, 0, idxs) 
        s_poses = torch.index_select(self.buffer_poses, 0, idxs)
        return s_imgs, s_poses
    
    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{self.opt.text}, {d} view"

                negative_text = f"{self.opt.negative}"

                # explicit negative dir-encoded text
                if self.opt.suppress_face:
                    if negative_text != '': negative_text += ', '

                    if d == 'back': negative_text += "face"
                    # elif d == 'front': negative_text += ""
                    elif d == 'side': negative_text += "face"
                    elif d == 'overhead': negative_text += "face"
                    elif d == 'bottom': negative_text += "face"
                
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file


    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        raise NotImplementedError()


    def evaluate(self, loader, name=None):
        raise NotImplementedError()


    def test(self, loader, save_path=None, name=None, write_video=True, idx = 0, shading = None):
        raise NotImplementedError()


    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = self.watermark.state_dict()

        state = {
            **state,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            # state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:
            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            state['model'] = self.model.state_dict()
            file_path = f"best_{name}.pth"
            torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict({(k):v for k,v in checkpoint_dict['model'].items()}, strict=False)

        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            # if 'mean_count' in checkpoint_dict:
            #     self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                self.model.verts *= checkpoint_dict['tet_scale'] / self.model.tet_scale
                self.model.tet_scale = checkpoint_dict['tet_scale']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")