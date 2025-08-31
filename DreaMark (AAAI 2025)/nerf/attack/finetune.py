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

_img2mse = lambda x, y : torch.mean((x - y) ** 2)
_mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

def _message_loss(fts, targets, m=1, loss_type='bce'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts/m), targets, reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

class FTAL(Attack):
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


    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.global_step < self.opt.albedo_iters+1:
            shading = 'albedo'
            ambient_ratio = 1.0
        else: 
            rand = random.random()
            if rand > 0.8: 
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4 and (not self.opt.no_textureless): 
                shading = 'textureless'
                ambient_ratio = 0.1
            else: 
                if not self.opt.no_lambertian:
                    shading = 'lambertian'
                    ambient_ratio = 0.1
                else:
                    shading = 'albedo'
                    ambient_ratio = 1.0                    

        # if random.random() < self.opt.p_normal:
        #     shading = 'normal'
        #     ambient_ratio = 1.0
        # 
        light_d = None
        if self.opt.normal:
            shading = 'normal'
            ambient_ratio = 1.0     
            if self.opt.p_textureless > random.random():
                shading = 'textureless'
                ambient_ratio = 0.1             
                light_d = data['rays_o'].contiguous().view(-1, 3)[0] + 0.3 * torch.randn(3, device=rays_o.device, dtype=torch.float)
                light_d = safe_normalize(light_d)             
        if self.global_step < self.opt.normal_iters+1:
            as_latent = True
            shading = 'normal'
            ambient_ratio = 1.0                   
        else:
            as_latent = False

        bg_color = None
        if self.global_step > 2000:
            if random.random() > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg
        
        if self.opt.backbone == "particle":
            self.model.mytraining = True
        binarize = False
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, light_d= light_d, perturb=False, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        ori_outputs = self.ori_model.render(rays_o, rays_d, mvp, H, W, staged=False, light_d=light_d, perturb=False, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        
        if self.opt.backbone == "particle":
            self.model.mytraining = False

        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        
        if as_latent:
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
            ori_pred_rgb = torch.cat([ori_outputs['images'], ori_outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3 if not self.opt.latent else 4).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
            ori_pred_rgb = ori_outputs['image'].reshape(B, H, W, 3 if not self.opt.latent else 4).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]

        loss = (pred_rgb - ori_pred_rgb).abs().mean() + (pred_rgb - ori_pred_rgb).abs().max()

        # regularizations
        if not self.opt.dmtet:
            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.opt.lambda_entropy > 0:

                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()

                # lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)

                loss = loss + self.opt.lambda_entropy * loss_entropy

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient
        else:
            if self.opt.lambda_normal > 0:
                loss = loss + self.opt.lambda_normal * outputs['normal_loss']

            if self.opt.lambda_lap > 0:
                loss = loss + self.opt.lambda_lap * outputs['lap_loss']

        return pred_rgb, pred_depth, loss, None, None, shading
    
    def post_train_step(self):

        if self.opt.backbone == 'grid':

            lambda_tv = min(1.0, self.global_step / 1000) * self.opt.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
        elif self.opt.backbone == "particle" and self.opt.lambda_tv > 0:
            self.scaler.unscale_(self.optimizer)
            self.model.encoders[self.model.idx].grad_total_variation(self.opt.lambda_tv, None, self.model.bound)       

    def eval_step(self, data, shading):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # shading = data['shading'] if 'shading' in data else 'albedo'
        # ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        # light_d = data['light_d'] if 'light_d' in data else None

        if shading == "albedo":
            ambient_ratio = 1.0
            light_d = None
        elif shading == "lambertian":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0]
            light_d = safe_normalize(light_d)
        elif shading == "textureless":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0] + 0.3 * torch.randn(3, device=rays_o.device, dtype=torch.float)
            light_d = safe_normalize(light_d)
            # light_d = None
        elif shading == "normal":
            ambient_ratio = 1.0
            light_d = None            
        else:
            raise NotImplementedError()

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        if not self.opt.latent:
            pred_rgb = outputs['image'].reshape(B, H, W, 3)
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 4)
            with torch.no_grad():
                pred_rgb = self.guidance.decode_latents(pred_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        # dummy 
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False, shading=None):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # if bg_color is not None:
        #     bg_color = bg_color.to(rays_o.device)
        # else:
        #     bg_color = torch.ones(3, device=rays_o.device) # [3]
        bg_color = torch.ones(3, device=rays_o.device)
        # shading = data['shading'] if 'shading' in data else 'albedo'
        # ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        # light_d = data['light_d'] if 'light_d' in data else None
        if shading == "albedo":
            ambient_ratio = 1.0
            light_d = None
        elif shading == "lambertian":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0]
            light_d = safe_normalize(light_d)
        elif shading == "textureless":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0]
            light_d = safe_normalize(light_d)
        elif shading == "normal":
            ambient_ratio = 1.0
            light_d = None            
        else:
            raise NotImplementedError()
    
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color)

        if not self.opt.latent:
            pred_rgb = outputs['image'].reshape(B, H, W, 3)
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 4)
            with torch.no_grad():
                pred_rgb = self.guidance.decode_latents(pred_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W) > 0.95

        return pred_rgb, pred_depth, pred_mask

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            if epoch >= self.opt.iter512 and self.opt.iter512 > 0:
                if epoch == self.opt.iter512:
                    print("Change into 512 resolution!")
                train_loader = self.train_loader512

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                pass
                # self.save_checkpoint(full=True, best=False)

            if self.epoch > 1000:
                continue

            if self.epoch % self.eval_interval == 0 or self.epoch < 2:
                self.evaluate_one_epoch(valid_loader, shading = "albedo")
                self.evaluate_one_epoch(valid_loader, shading = "normal")
                self.evaluate_one_epoch(valid_loader, shading = "textureless")
                if not self.opt.albedo or self.opt.p_normal > 0:           
                    self.evaluate_one_epoch(valid_loader, shading = "lambertian")
                if self.epoch < 402:
                    self.save_checkpoint(full=False, best=False)
                    # self.save_checkpoint(full=False, best=True)

            unet_bs = 8 if not self.opt.lora else 2

            if self.epoch % self.opt.test_interval == 0:
                self.save_checkpoint(full=False, best=True)
                if self.opt.backbone == 'particle':
                    for idx in range(self.opt.n_particles):
                        self.model.set_idx(idx)
                        for shading in ["textureless", "albedo", "normal"]:
                            self.test(self.test_loader, idx=idx, shading = shading)   
                        # break 
                else:
                    self.test(self.test_loader)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        raise NotImplementedError()

    def test(self, loader, save_path=None, name=None, write_video=True, idx = 0, shading = None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data, shading=shading)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)
                pred_depth = cv2.applyColorMap(pred_depth, cv2.COLORMAP_JET)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                # else:
                    # if i % 3 == 0:
                    #     cv2.imwrite(os.path.join(save_path, f'img_{name}_{idx:02d}_{i:06d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    #     cv2.imwrite(os.path.join(save_path, f'img_{name}_{idx:02d}_{i:06d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            
            imageio.mimwrite(os.path.join(save_path, f'{name}_{idx:02d}_{shading}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            if shading == "albedo":
                imageio.mimwrite(os.path.join(save_path, f'{name}_{idx:02d}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        msg_length = self.watermark.msg.shape[-1]
        random_msg = torch.zeros_like(self.watermark.msg)
        for data in loader:
            self.model.set_idx()
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss, pseudo_loss, latents, shading = self.train_step(data)

            fts = self.watermark.decoder(pred_rgbs)[:, :msg_length]
            loss += _message_loss(fts=fts, targets=random_msg)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.scheduler_update_every_step:
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None, shading = None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()
        if shading == 'albedo':
            acc, pred_rgbs = self.watermark.eval_step(self.model)
            _, rgb_target = self.watermark.eval_step(self.ori_model)
            img_mse = _img2mse(pred_rgbs[0], rgb_target[0])
            img_PSNR = _mse2psnr(img_mse)
            self.log(f'Acc={acc}, PSNR={img_PSNR}')
            os.makedirs(os.path.join(self.workspace, 'validation'), exist_ok=True)
            for i, pred_rgb in enumerate(pred_rgbs):
                torchvision.utils.save_image(pred_rgb, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:06d}' + "-mark-"+shading+f"{i}.png"), nrow=self.opt.val_size, normalize=True, range=(0,1))


        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            pre_imgs = None
            pre_depths = None
            for idx in range(self.opt.val_nz):
                if self.opt.backbone == 'particle':
                    self.model.set_idx(idx)
                for data in loader:    
                    self.local_step += 1

                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, loss = self.eval_step(data, shading)

                    if self.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / self.world_size
                        
                        preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_list, preds)
                        preds = torch.cat(preds_list, dim=0)

                        preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_depth_list, preds_depth)
                        preds_depth = torch.cat(preds_depth_list, dim=0)
                    
                    loss_val = loss.item()
                    total_loss += loss_val

                    if self.local_rank == 0:
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        if pre_imgs == None:
                            pre_imgs = preds
                        else:
                            pre_imgs = torch.cat([pre_imgs, preds], dim = 0)
                        if pre_depths == None:
                            pre_depths = preds_depth
                        else:
                            pre_depths = torch.cat([pre_depths, preds_depth], dim = 0)

                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                        pbar.update(loader.batch_size)
                if not (self.opt.backbone == 'particle'):
                    break
            if self.local_rank == 0:
                torchvision.utils.save_image(pre_imgs.permute(0,3,1,2), os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:06d}' + "-rgb-"+shading+".png"), nrow=self.opt.val_size, normalize=True, range=(0,1))
                torchvision.utils.save_image(pre_imgs.permute(0,3,1,2)[0], os.path.join(self.workspace, "validation", f'{self.name}_ep{self.epoch:06d}' + "-rgb-"+shading+"0.png"), normalize=True, range=(0,1))
                if shading == "albedo":
                    torchvision.utils.save_image(pre_depths.unsqueeze(1), os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:06d}' + "-depth.png"), nrow=self.opt.val_size, normalize=True)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

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