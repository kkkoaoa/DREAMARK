from model.wm_util import *
from model.signed_distance_function import *

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from typing import List, Callable, Iterable

from util import *
from backend import *

mlconfig.register(optim.lr_scheduler.MultiStepLR)
mlconfig.register(optim.lr_scheduler.StepLR)
mlconfig.register(optim.lr_scheduler.ExponentialLR)
mlconfig.register(optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(optim.lr_scheduler.ReduceLROnPlateau)

########################################################################################
#                                      Base                                            #
########################################################################################

def siren_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)

class LinearRelu(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=bias),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class ConvBNRelu1D(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBNRelu2D(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class WeightNet(nn.Module):
    def __init__(self, mlp_channels=[64, 64, 64]):
        super().__init__()
        in_channel = 3
        self.mlps = nn.Sequential()
        for idx, channel in enumerate(mlp_channels):
            self.mlps.append(
                nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=1) \
                    if idx == len(mlp_channels) - 1 else \
                ConvBNRelu2D(in_channels=in_channel, out_channels=channel, kernel_size=1)
            )
            in_channel = channel

    def forward(self, offset):
        return self.mlps(offset)


########################################################################################
#                                       Layer                                          #
########################################################################################

class LipBoundedPosEnc(nn.Module):

    def __init__(self, inp_features, n_freq, cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat
        if self.cat_inp:
            self.out_dim += self.inp_feat

    def forward(self, x):
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        out = torch.cat(
            [sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq)
        const_norm = torch.cat(
            [const, const], dim=-1).view(
            1, 1, 1, self.n_freq * 2).expand(
            -1, -1, self.inp_feat, -1).reshape(
            1, 1, 2 * self.inp_feat * self.n_freq)

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            const_norm = torch.cat(
                [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1)

            return out / const_norm / np.sqrt(self.n_freq * 2 + 1)
        else:

            return out / const_norm / np.sqrt(self.n_freq * 2)

########################################################################################
#                                      Block                                           #
########################################################################################

class InvertibleResBlockLinear(nn.Module):
    def __init__(self, inp_dim, hid_dim, nblocks=1,
                 nonlin='leaky_relu',
                 pos_enc_freq=None):
        super().__init__()
        self.dim = inp_dim
        self.nblocks = nblocks

        self.pos_enc_freq = pos_enc_freq
        if self.pos_enc_freq is not None:
            inp_dim_af_pe = self.dim * (self.pos_enc_freq * 2 + 1)
            self.pos_enc = LipBoundedPosEnc(self.dim, self.pos_enc_freq)
        else:
            self.pos_enc = lambda x: x
            inp_dim_af_pe = inp_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.utils.spectral_norm(
                nn.Linear(inp_dim_af_pe, hid_dim)
            )
        )
        for _ in range(self.nblocks):
            self.blocks.append(
                nn.utils.spectral_norm(
                    nn.Linear(hid_dim, hid_dim),
                )
            )
        self.blocks.append(
            nn.utils.spectral_norm(
                nn.Linear(hid_dim, self.dim),
            )
        )

        self.nonlin = nonlin.lower()
        if self.nonlin == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif self.nonlin == 'relu':
            self.act = nn.ReLU()
        elif self.nonlin == 'elu':
            self.act = nn.ELU()
        elif self.nonlin == 'softplus':
            self.act = nn.Softplus()
        elif self.nonlin == "sine":
            self.act = Sine(30)
            self.blocks.apply(siren_sine_init)
            self.blocks[0].apply(first_layer_sine_init)
        else:
            raise NotImplementedError

    def forward_g(self, x):
        orig_dim = len(x.size())
        if orig_dim == 2:
            x = x.unsqueeze(0)

        y = self.pos_enc(x)
        for block in self.blocks[:-1]:
            y = self.act(block(y))
        y = self.blocks[-1](y)

        if orig_dim == 2:
            y = y.squeeze(0)

        return y

    def forward(self, x):
        return x + self.forward_g(x)

    def invert(self, y, verbose=False, iters=15):
        return fixed_point_invert(
            lambda x: self.forward_g(x), y, iters=iters, verbose=verbose
        )

@mlconfig.register
class FCBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, n_blocks):
        super().__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Linear(in_channel, hidden_channel))
        self.net.append(Sine(30))
        for _ in range(n_blocks):
            self.net.append(nn.Linear(hidden_channel, hidden_channel))
            self.net.append(Sine(30))
        self.net.append(nn.Linear(hidden_channel, out_channel))
        
        self.net.apply(siren_sine_init)
        self.net[0].apply(first_layer_sine_init)
    
    def forward(self, x):
        return self.net(x)

# @mlconfig.register
# class FCBlock(nn.Module):
#     def __init__(self, in_channel, hidden_channel, out_channel, n_blocks):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_channel, hidden_channel), Sine(30),
#             nn.Linear(hidden_channel, hidden_channel), Sine(30),
#             nn.Linear(hidden_channel, hidden_channel), Sine(30),
#             nn.Linear(hidden_channel, out_channel)
#         )
#         self.net.apply(siren_sine_init)
#         self.net[0].apply(first_layer_sine_init)
    
#     def forward(self, x):
#         return self.net(x)

########################################################################################
#                                      Module                                          #
########################################################################################

class DeformNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.dim = 3
        self.out_dim = 3
        self.hidden_size = 256
        self.n_blocks = 6
        self.n_g_blocks = 1

        # Network modules
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(
                InvertibleResBlockLinear(
                    self.dim, self.hidden_size,
                    nblocks=self.n_g_blocks, nonlin="elu",
                    pos_enc_freq=5,
                )
            )

    def forward(self, x):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out

    def invert(self, y, verbose=False, iters=15):
        x = y
        for block in self.blocks[::-1]:
            x = block.invert(x, verbose=verbose, iters=iters)
        return x


class DeformationWrapper(nn.Module):
    def __init__(self, orig, deform):
        super().__init__()
        self.orig = orig
        self.deform = deform

    def forward(self, x):
        x_deform = x
        x_deform = self.deform(x)

        out = self.orig(x_deform)
        return x_deform, out


def fixed_point_invert(g, y, iters=15, verbose=False):
    with torch.no_grad():
        x = y
        dim = x.size(-1)
        for i in range(iters):
            x = y - g(x)
            if verbose:
                err = (y - (x + g(x))).view(-1, dim).norm(dim=-1).mean()
                err = err.detach().cpu().item()
                print("iter:%d err:%s" % (i, err))
    return x