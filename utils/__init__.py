from .cfgnode import CfgNode
from .rend_util import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from glob import glob
import os
import numpy as np
from pytorch_lightning.callbacks import RichProgressBar
from rich.progress import TextColumn

class RichProgressBarWithScanId(RichProgressBar):
    def __init__(self, scan_id, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.custom_column = TextColumn(f"[progress.description]scan_id: {scan_id}")
    
    def configure_columns(self, trainer):
        return super().configure_columns(trainer) + [self.custom_column]


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def glob_depths(path):
    imgs = []
    for ext in ['*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

glob_normal = glob_depths

def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels, device=model_input['uv'].device), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split


def split_dict(d, batch_size=10000):
    keys = d.keys()
    splits = {}
    for k in d:
        splits[k] = torch.split(d[k], batch_size)
        n_splits = len(splits[k])
    split_inputs = []
    for i in range(n_splits):
        split = {}
        for k in d:
            split[k] = splits[k][i]
        split_inputs.append(split)
    return split_inputs



def detach_dict(d):
    return {k: v.detach() for k, v in d.items() if torch.is_tensor(v)}


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs


def merge_dict(dicts):
    output = {}
    for entry in dicts[0]:
        output[entry] = torch.cat([r[entry] for r in dicts], dim=0)
    return output

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

def kmeans_pp_centroid(points: torch.Tensor, k):
    n, c = points.shape
    centroids = torch.zeros(k, c, device=points.device)
    centroids[0, :] = points[np.random.randint(0, n), :].clone()
    d = [0.0] * n
    for i in range(1, k):
        sum_all = 0
        d = (points.unsqueeze(1) - centroids[:i,:].unsqueeze(0)).norm(p=2, dim=-1).min(dim=1).values
        sum_all = d.sum() * np.random.random()
        cumsum = torch.cumsum(d, dim=0)
        j = ((cumsum - sum_all) > 0).int().argmax()
        centroids[i,:] = points[j,:].clone()
    return centroids