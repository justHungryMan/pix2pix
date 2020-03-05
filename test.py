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
import tqdm

from utils import *
from network import Generator, Discriminator


class image_preprocessing(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))
        ])
        self.dir_data = os.path.join(root_dir, 'val')
        self.image = glob.glob(os.path.join(root_dir, 'val') + '/*.jpg')

    def __getitem__(self, idx):
        AB_path = self.image[idx]
        AB = Image.open(AB_path)

        AB = self.transforms(AB)
        # 3 * 256 * 512
        _, h, w = AB.shape
        A = AB.clone().detach()[:, :, :int(w/2)]
        B = AB.clone().detach()[:, :, int(w/2):]
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image)

def main():
    test_image_dataset = image_preprocessing(args.dataset)
    data_loader = DataLoader(test_image_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    G = Generator()
    G.load_state_dict(torch.load(args.model_path))
    G.eval()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if torch.cuda.is_available():
        G = G.cuda()

    for step, data in enumerate(data_loader):
        real_A = to_variable(data['A'])
        real_B = to_variable(data['B'])
        fake_B = G(real_A)

        batch_image = torch.cat((torch.cat((real_A, fake_B), 3), real_B), 3)
        for i in range(args.batch_size):
            torchvision.utils.save_image(denorm(batch_image[i]), args.save_path + 'result_{step}_undenorm.jpg'.format(step=step * 4 + i))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch pix2pix implementation')

    parser.add_argument('--dataset', default='./data/datasets/edges2handbags/', type=str, help='dataset path')
    parser.add_argument('--save_path', default='./result/', type=str, help='save path')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')

    args = parser.parse_args()

    main()