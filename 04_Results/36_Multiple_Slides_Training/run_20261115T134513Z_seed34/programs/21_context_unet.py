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
    @torch.inference_mode()
    def predict_probabilities(self,x):
        """Match training's 256px field size; blend overlapping image-only views.

        GroupNorm depends on the spatial field. Evaluating the entire 1024px
        image in a single call changes that field versus 256px training crops.
        Fixed overlap blending avoids seams and uses no reference labels.
        """
        assert x.shape[0]==1
        outputs=[]
        weight=torch.hann_window(256,periodic=False,device=x.device)
        weight=(weight[:,None]*weight[None,:]).clamp_min(.05)[None,None]
        for scale in [1.,.5]:
            z=F.interpolate(x,scale_factor=scale,mode='bilinear',align_corners=False) if scale!=1 else x
            h,w=z.shape[-2:];z=F.pad(z,(64,64,64,64),mode='reflect');hh,ww=z.shape[-2:]
            ys=list(range(0,hh-255,128));xs=list(range(0,ww-255,128))
            if ys[-1]!=hh-256:ys.append(hh-256)
            if xs[-1]!=ww-256:xs.append(ww-256)
            positions=[(y,t) for y in ys for t in xs];total=torch.zeros((1,3,hh,ww),device=x.device);denom=torch.zeros((1,1,hh,ww),device=x.device)
            for start in range(0,len(positions),8):
                batch=positions[start:start+8];tiles=torch.cat([z[:,:,y:y+256,t:t+256] for y,t in batch]);p=self(tiles).softmax(1)
                for j,(y,t) in enumerate(batch):total[:,:,y:y+256,t:t+256]+=p[j:j+1]*weight;denom[:,:,y:y+256,t:t+256]+=weight
            p=(total/denom.clamp_min(1e-6))[:,:,64:64+h,64:64+w]
            outputs.append(F.interpolate(p,size=x.shape[-2:],mode='bilinear',align_corners=False))
        return torch.stack(outputs).mean(0)

def choose_device(requested='auto'):
    return torch.device('mps' if requested!='cpu' and torch.backends.mps.is_available() else 'cpu')
