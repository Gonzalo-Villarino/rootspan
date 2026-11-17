"""Compact four-class U-Net, trained from scratch; no published PAT weights."""
import torch
from torch import nn
import torch.nn.functional as F
class Block(nn.Module):
 def __init__(self,a,b):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
 def forward(self,x):return self.net(x)
class SmallUNet(nn.Module):
 def __init__(self):
  super().__init__();self.down=nn.ModuleList([Block(3,16),Block(16,32),Block(32,64),Block(64,128),Block(128,256)]);self.up=nn.ModuleList([Block(256+128,128),Block(128+64,64),Block(64+32,32),Block(32+16,16)]);self.head=nn.Conv2d(16,4,1)
 def forward(self,x):
  skips=[]
  for i,b in enumerate(self.down):
   if i:x=F.max_pool2d(x,2)
   x=b(x);skips.append(x)
  for b,skip in zip(self.up,skips[-2::-1]):x=b(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
  return self.head(x)
