import math
import random
from collections import OrderedDict

import numpy as np
import swin
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import init


class ResBlock(nn.Module):
    def __init__(self, inc, midc):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn1 = nn.GroupNorm(16, midc)
        self.conv2 = nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(16, midc)
        self.conv3 = nn.Conv2d(midc, inc, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x + x_
        x = self.relu(x)
        return x


class AEALblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False, width=5):
        super(AEALblock, self).__init__()
        self.self_attn2 = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, batch_first=batch_first)
        self.self_attn1 = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.width = width
        self.trans=nn.Sequential(nn.Conv2d(d_model+512,d_model//2,1,1,0),ResBlock(d_model//2,d_model//4),nn.Conv2d(d_model//2,d_model,1,1,0))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, src, feats, ):
        src= self.gamma* self.trans(torch.cat([src,feats],1))+src
        b, c, h, w = src.shape
        x1 = src[:, 0:c // 2]
        x1_ = rearrange(x1, 'b c (h1 h2) w -> b c h1 h2 w', h2=self.width)
        x1_ = rearrange(x1_, 'b c h1 h2 w -> (b h1) (h2 w) c')
        x2 = src[:, c // 2:]
        x2_ = rearrange(x2, 'b c h (w1 w2) -> b c h w1 w2', w2=self.width)
        x2_ = rearrange(x2_, 'b c h w1 w2 -> (b w1) (h w2) c')
        x = rearrange(src, 'b c h w-> b (h w) c')
        x = self.norm1(x + self._sa_block(x1_, x2_, h, w))
        x = self.norm2(x + self._ff_block(x))
        x = rearrange(x, 'b (h w) c->b c h w', h=h, w=w)
        return x

    def _sa_block(self, x1, x2, h, w):
        x1 = self.self_attn1(x1, x1, x1,
                             attn_mask=None,
                             key_padding_mask=None,
                             need_weights=False)[0]

        x2 = self.self_attn2(x2, x2, x2,
                             attn_mask=None,
                             key_padding_mask=None,
                             need_weights=False)[0]

        x1 = rearrange(x1, '(b h1) (h2 w) c-> b (h1 h2 w) c', h2=self.width, h1=h // self.width)
        x2 = rearrange(x2, ' (b w1) (h w2) c-> b (h w1 w2) c', w2=self.width, w1=w // self.width)
        x = torch.cat([x1, x2], dim=2)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AEMatter(nn.Module):
    def __init__(self):
        super(AEMatter, self).__init__()
        trans = swin.SwinTransformer(pretrain_img_size=224,
                                     embed_dim=96,
                                     depths=[2, 2, 6, 2],
                                     num_heads=[3, 6, 12, 24],
                                     window_size=7,
                                     ape=False,
                                     drop_path_rate=0.0,
                                     patch_norm=True,
                                     use_checkpoint=False)
        trans.patch_embed.proj = nn.Conv2d(64, 96, 3, 2, 1)
        self.start_conv0 = nn.Sequential(nn.Conv2d(6, 48, 3, 1, 1), nn.PReLU(48))
        self.start_conv = nn.Sequential(nn.Conv2d(48, 64, 3, 2, 1), nn.PReLU(64), nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.PReLU(64))
        self.trans = trans
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=640 + 768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 384, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True), )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=True), )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192 + 96, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True), )
        self.ctran0 = swin.BasicLayer(256, 3, 8, 7, drop_path=0.0)
        self.ctran1 = swin.BasicLayer(256, 3, 8, 7, drop_path=0.0)
        self.ctran2 = swin.BasicLayer(192, 3, 6, 7, drop_path=0.0)
        self.ctran3 = swin.BasicLayer(128, 3, 4, 7, drop_path=0.0)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(64),
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(48))
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=48 + 48 + 6, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(32), nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(32), nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upn = nn.Upsample(scale_factor=2, mode='nearest')
        self.apptrans = nn.Sequential(nn.Conv2d(256 + 384, 256, 1, 1, bias=True), ResBlock(256, 128),
                                      ResBlock(256, 128), nn.Conv2d(256, 512, 2, 2, bias=True), ResBlock(512, 128))
        self.emb = nn.Sequential(nn.Conv2d(768, 640, 1, 1, 0), ResBlock(640, 160))
        self.embdp = nn.Sequential(nn.Conv2d(640, 640, 1, 1, 0))
        self.h2l = nn.Conv2d(768, 256, 1, 1, 0)
        self.width = 5
        self.trans1 = AEALblock(d_model=640, nhead=20, dim_feedforward=2048, dropout=0.0, width=self.width)
        self.trans2 = AEALblock(d_model=640, nhead=20, dim_feedforward=2048, dropout=0.0, width=self.width)
        self.trans3 = AEALblock(d_model=640, nhead=20, dim_feedforward=2048, dropout=0.0, width=self.width)

    def aeal(self, x, sem):
        xe = self.emb(x)
        x_ = xe
        x_ = self.embdp(x_)
        b, c, h1, w1 = x_.shape
        bnew_ph = int(np.ceil(h1 / self.width) * self.width) - h1
        bnew_pw = int(np.ceil(w1 / self.width) * self.width) - w1
        newph1 = bnew_ph // 2
        newph2 = bnew_ph - newph1
        newpw1 = bnew_pw // 2
        newpw2 = bnew_pw - newpw1
        x_ = F.pad(x_, (newpw1, newpw2, newph1, newph2))
        sem = F.pad(sem, (newpw1, newpw2, newph1, newph2))
        x_ = self.trans1(x_, sem)
        x_ = self.trans2(x_, sem)
        x_ = self.trans3(x_, sem)
        x_ = x_[:, :, newph1:h1 + newph1, newpw1:w1 + newpw1]
        return x_

    def forward(self, x, y):
        inputs = torch.cat((x, y), 1)
        x = self.start_conv0(inputs)
        x_ = self.start_conv(x)
        x1, x2, x3, x4 = self.trans(x_)
        x4h = self.h2l(x4)
        x3s = self.apptrans(torch.cat([x3, self.upn(x4h)], 1))
        x4_ = self.aeal(x4, x3s)
        x4 = torch.cat((x4, x4_), 1)
        X4 = self.conv1(x4)
        wh, ww = X4.shape[2], X4.shape[3]
        X4 = rearrange(X4, 'b c h w -> b (h w) c')
        X4, _, _, _, _, _ = self.ctran0(X4, wh, ww)
        X4 = rearrange(X4, 'b (h w) c -> b c h w', h=wh, w=ww)
        X3 = self.up(X4)
        X3 = torch.cat((x3, X3), 1)
        X3 = self.conv2(X3)
        wh, ww = X3.shape[2], X3.shape[3]
        X3 = rearrange(X3, 'b c h w -> b (h w) c')
        X3, _, _, _, _, _ = self.ctran1(X3, wh, ww)
        X3 = rearrange(X3, 'b (h w) c -> b c h w', h=wh, w=ww)
        X2 = self.up(X3)
        X2 = torch.cat((x2, X2), 1)
        X2 = self.conv3(X2)
        wh, ww = X2.shape[2], X2.shape[3]
        X2 = rearrange(X2, 'b c h w -> b (h w) c')
        X2, _, _, _, _, _ = self.ctran2(X2, wh, ww)
        X2 = rearrange(X2, 'b (h w) c -> b c h w', h=wh, w=ww)
        X1 = self.up(X2)
        X1 = torch.cat((x1, X1), 1)
        X1 = self.conv4(X1)
        wh, ww = X1.shape[2], X1.shape[3]
        X1 = rearrange(X1, 'b c h w -> b (h w) c')
        X1, _, _, _, _, _ = self.ctran3(X1, wh, ww)
        X1 = rearrange(X1, 'b (h w) c -> b c h w', h=wh, w=ww)
        X0 = self.up(X1)
        X0 = torch.cat((x_, X0), 1)
        X0 = self.conv5(X0)
        X = self.up(X0)
        X = torch.cat((inputs, x, X), 1)
        alpha = self.convo(X)
        alpha = torch.clamp(alpha, min=0, max=1)
        return alpha
