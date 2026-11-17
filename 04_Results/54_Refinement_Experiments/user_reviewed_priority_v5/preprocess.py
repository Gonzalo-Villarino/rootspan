"""Fixed image-only contrast transform, applied equally to teaching and prediction.
No reference mask, dataset statistics or tissue labels are used.
"""
import numpy as np
from skimage.exposure import equalize_adapthist
def prepare(rgb,mode='none'):
 if mode=='none':return rgb
 assert mode=='grayscale_clahe'
 g=rgb[:,:,1].astype('float32')/255
 a=equalize_adapthist(g,kernel_size=64,clip_limit=.01,nbins=256)
 return np.repeat(np.round(a*255).astype('uint8')[:,:,None],3,axis=2)
