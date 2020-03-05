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
    
    denormalize = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    return denormalize(x).clamp(0, 1)
