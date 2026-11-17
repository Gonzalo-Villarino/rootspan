"""Wider-context learning U-Net; not PAT's pretrained UNet++.

Four pooling levels plus a dilated bottom block give more cellular context.
Residual blocks and GroupNorm make optimization less dependent on minibatch
composition. Input RGB/255, outputs P/E/B logits. No coordinates or masks input.
"""
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,ci,co,dilation=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(ci,co,3,padding=dilation,dilation=dilation,bias=False),nn.GroupNorm(4,co),nn.SiLU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(4,co))
        self.skip=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
    def forward(self,x):return F.silu(self.body(x)+self.skip(x))

class ThreeClassUNet(nn.Module):
    multiscale=True
    def __init__(self):
        super().__init__();self.enc=nn.ModuleList([Block(3,16),Block(16,32),Block(32,64),Block(64,128)]);self.bottom=Block(128,256,dilation=2)
        self.dec=nn.ModuleList([Block(256+128,128),Block(128+64,64),Block(64+32,32),Block(32+16,16)]);self.output=nn.Conv2d(16,3,1)
    def forward(self,x):
        skips=[]
        for b in self.enc:x=b(x);skips.append(x);x=F.max_pool2d(x,2)
        x=self.bottom(x)
        for b,s in zip(self.dec,reversed(skips)):x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False);x=b(torch.cat([x,s],1))
        return self.output(x)

def choose_device(requested='auto'):
    return torch.device('mps' if requested!='cpu' and torch.backends.mps.is_available() else 'cpu')
