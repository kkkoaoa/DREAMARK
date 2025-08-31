import torch
import torch.nn.functional as F
import numpy as np
import random
import rsa

from .utils import safe_normalize, get_rays
from .provider import circle_poses

from hidden import *

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

class Mark:
    def __init__(self, opt, device, H, W, trigger_size=100):
        self.opt = opt
        self.device = device
        self.H = H
        self.W = W
        self.cx = H / 2
        self.cy = W / 2
        self.near = opt.min_near
        self.far = 1000
        self.trigger_size = trigger_size
        self.phis = [index/trigger_size*360 for index in range(trigger_size)]

        self.radius = opt.val_radius
        self.theta = opt.val_theta
        self.angle_overhead = opt.angle_overhead
        self.angle_front = opt.angle_front

    def collate(self, index):
        # circle pose
        phi = self.phis[index]
        #poses, dirs = circle_poses(self.device, radius=self.opt.radius_range[1] * 1.2, theta=self.opt.val_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
        poses, dirs = circle_poses(self.device, radius=self.opt.val_radius, theta=self.opt.val_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

        # fixed focal
        fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0], 
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]
        
        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'pose': poses,
            'mvp': mvp,
        }
        # print(self.H, rays['rays_o'].shape)
        return data
    
    def render(self, model, i):
        data = self.collate(i)

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

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

        as_latent = False

        # bg_color = torch.zeros(3, device=rays_o.device)
        bg_color = None
        
        if self.opt.backbone == "particle":
            model.mytraining = True
        binarize = False
        outputs = model.render(rays_o, rays_d, mvp, H, W, staged=False, light_d=light_d, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        if self.opt.backbone == "particle":
            model.mytraining = False

        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        
        if as_latent:
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3 if not self.opt.latent else 4).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        
        return pred_rgb
    
    def state_dict(self):
        raise NotImplementedError
    
    def load_state_dict(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
    
    def eval_step(self):
        raise NotImplementedError



class ResidueMark(Mark):
    def __init__(self, opt, device, H, W, guidance, trigger_size=100,
                 objective_size=[256, 253], threshold=0.1, lamda=1, divider=0, **kwargs):
        super().__init__(opt, device, H, W, trigger_size)

        self.copyright = b'This work is done by Peking University'

        signature_set = bytes()
        clients = 1  # if embed else 0

        for i in range(clients):
            (pubkey, privkey) = rsa.newkeys(512)
            message = self.copyright
            signature = rsa.sign(message, privkey, 'SHA-256')
            signature_set += signature
        signature_set = rsa.compute_hash(signature_set, 'SHA-256')
        b_sig = list(bin(int(signature_set.hex(), base=16)).lstrip('0b'))  # hex -> bin
        b_sig = list(map(int, b_sig))  # bin-> int
        while len(b_sig) < 256:
            b_sig.insert(0, 0)
        self.sig = torch.tensor([-1 if i == 0 else i for i in b_sig], dtype=torch.float, device=device)

        self.guidance = guidance
        self.objective_size = objective_size
        self.threshold = threshold
        self.lamda = lamda
        self.divider = divider
    
    def extract_weight(self, latents):
        latents = latents.view(-1)[:latents.numel() // self.objective_size[1] * self.objective_size[1]]
        latents = F.adaptive_avg_pool1d(latents[None, None],
                                                 self.objective_size[0] * self.objective_size[1]).squeeze(0).view(self.objective_size)
        return latents
    
    def construct_residual(self, weight_extraction):
        if self.divider != 0:
            idx = torch.argsort(torch.abs(weight_extraction),
                                dim=1)[:, :int(self.objective_size[1] / self.divider + 0.5)]
            for i in range(self.objective_size[0]):
                weight_extraction[i, idx[i]] = 0

        return torch.mean(weight_extraction, dim=1)
    
    def train_step(self, model):
        i = np.random.randint(0, self.trigger_size)

        pred_rgb = self.render(model, i)

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        latents = self.guidance.encode_imgs(pred_rgb_512)
        extracted_weights = self.extract_weight(latents)
        pred_raw_sig = self.construct_residual(extracted_weights)

        return self.lamda * F.relu(self.threshold - self.sig.view(-1) * pred_raw_sig.view(-1)).sum()
    
    @torch.no_grad()
    def eval_step(self, model):
        final_acc = []
        final_pred_rgb = []
        for i in range(self.trigger_size):
            pred_rgb = self.render(model, i)
            final_pred_rgb.append(pred_rgb)
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            latents = self.guidance.encode_imgs(pred_rgb_512)
            extracted_weights = self.extract_weight(latents)
            pred_raw_sig = self.construct_residual(extracted_weights)
            
            acc = torch.as_tensor(torch.sign(pred_raw_sig) == self.sig).float().mean().item()
            final_acc.append(acc)
        return final_acc, final_pred_rgb
    
    @torch.no_grad()
    def eval_img(self, rgb):
        rgb_512 = F.interpolate(rgb, (512, 512), mode="bilinear", align_corners=False)
        latents = self.guidance.encode_imgs(rgb_512)
        extracted_weights = self.extract_weight(latents)
        pred_raw_sig = self.construct_residual(extracted_weights)

        return torch.as_tensor(torch.sign(pred_raw_sig) == self.sig).float().mean().item()
    
    def state_dict(self):
        return {
            "sig": self.sig,
            "phis": self.phis,
            "radius": self.radius,
            "theta": self.theta,
            "angle_overhead": self.angle_overhead,
            "angle_front": self.angle_front,
            "objective_size": self.objective_size,
            "divider": self.divider,
        }
    
    def load_state_dict(self, checkpoint_dict):
        if 'sig' in checkpoint_dict.keys():
            self.sig = checkpoint_dict['sig']
        if 'phis' in checkpoint_dict.keys():
            self.trigger_size = len(checkpoint_dict['phis'])
            self.phis = checkpoint_dict['phis']
        if 'radius' in checkpoint_dict.keys():
            self.radius = checkpoint_dict['radius']
        if 'theta' in checkpoint_dict.keys():
            self.theta = checkpoint_dict['theta']
        if 'angle_overhead' in checkpoint_dict.keys():
            self.angle_overhead = checkpoint_dict['angle_overhead']
        if 'angle_front' in checkpoint_dict.keys():
            self.angle_front = checkpoint_dict['angle_front']
        if 'objective_size' in checkpoint_dict.keys():
            self.objective_size = checkpoint_dict['objective_size']
        if 'divider' in checkpoint_dict.keys():
            self.divider = checkpoint_dict['divider']


class HiddenMark(Mark):
    def __init__(self, opt, device, H, W, trigger_size=100, msg_length=48, **kwargs):
        super().__init__(opt, device, H, W, trigger_size)

        assert msg_length <= 48
        self.msg = torch.FloatTensor(np.random.choice([0,1], size=(1, msg_length))).to(device)
        
        encoder = HiddenEncoder(num_blocks=4, num_bits=48, channels=64)
        decoder = HiddenDecoder(num_blocks=8, num_bits=48, channels=64)
        hidden = EncoderDecoder(encoder=encoder, decoder=decoder, attenuation=None, augmentation=None, scale_channels=False, scaling_i=1, scaling_w=1, num_bits=msg_length, redundancy=1).to(device)
        checkpoint = torch.load("ckpt/hidden_replicate.pth", map_location=device)
        new_ckpt = {}
        for k, v in checkpoint['encoder_decoder'].items():
            if k.startswith('module.'):
                new_ckpt[k[7:]] = v
            else:
                new_ckpt[k] = v
        hidden.load_state_dict(new_ckpt)
        self.decoder = decoder


    def state_dict(self):
        return {
            "msg": self.msg,
            "phis": self.phis,
            "radius": self.radius,
            "theta": self.theta,
            "angle_overhead": self.angle_overhead,
            "angle_front": self.angle_front,
        }


    def load_state_dict(self, checkpoint_dict):
        if 'msg' in checkpoint_dict.keys():
            self.msg = checkpoint_dict['msg']
        if 'phis' in checkpoint_dict.keys():
            self.trigger_size = len(checkpoint_dict['phis'])
            self.phis = checkpoint_dict['phis']
        if 'radius' in checkpoint_dict.keys():
            self.radius = checkpoint_dict['radius']
        if 'theta' in checkpoint_dict.keys():
            self.theta = checkpoint_dict['theta']
        if 'angle_overhead' in checkpoint_dict.keys():
            self.angle_overhead = checkpoint_dict['angle_overhead']
        if 'angle_front' in checkpoint_dict.keys():
            self.angle_front = checkpoint_dict['angle_front']

    def train_step(self, model):
        msg_length = self.msg.shape[-1]
        i = np.random.randint(0, self.trigger_size)

        pred_rgb = self.render(model, i)
        
        fts = self.decoder(pred_rgb)[:, :msg_length]
        return _message_loss(fts=fts, targets=self.msg)
    
    
    @torch.no_grad()
    def eval_step(self, model):
        # self.trigger_size=trigger_size=100
        # self.phis = [index/trigger_size*360 for index in range(trigger_size)]
        msg_length = self.msg.shape[-1]
        final_acc = []
        pred_rgbs = []
        for i in range(self.trigger_size):
            pred_rgb = self.render(model, i)
            pred_rgbs.append(pred_rgb)

            fts = self.decoder(pred_rgb)[:, :msg_length]
            acc = (fts.round().clip(0,1)==self.msg).float().mean().item()
            final_acc.append(acc)
        return final_acc, pred_rgbs

    @torch.no_grad()
    def eval_img(self, rgb):
        msg_length = self.msg.shape[-1]
        fts = self.decoder(rgb)[:, :msg_length]
        acc = (fts.round().clip(0,1)==self.msg).float().mean().item()
        return acc