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
    x = x / 2 + 0.5
    return x.clamp(0, 1)
