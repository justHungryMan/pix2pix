import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import argparse
import glob
import os
import datetime

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
def denorm(x):
    
    return ((x + 1) / 2).clamp(0, 1)

def load_ckp(checkpoint_fpath, G, D, G_optimizer, D_optimizer):
    checkpoint = torch.load(checkpoint_fpath)

    G.load_state_dict(checkpoint['G_model'])
    D.load_state_dict(checkpoint['D_model'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])
    epoch = checkpoint['epoch']

    return G, D, G_optimizer, D_optimizer, epoch