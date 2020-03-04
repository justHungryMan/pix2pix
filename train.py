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
        self.dir_data = os.path.join(root_dir, 'train')
        self.image = glob.glob(os.path.join(root_dir, 'train') + '/*.jpg')

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

def patch_loss(criterion, input, TF):
    if TF is True:
        comparison = torch.ones_like(input)
    else:
        comparison = torch.zeros_like(input)
    return criterion(input, comparison)


def main():
    train_image_dataset = image_preprocessing(args.dataset)
    data_loader = DataLoader(train_image_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    
    criterion = nn.BCELoss()
    euclidean_l1 = nn.L1Loss()
    G = Generator()
    D = Discriminator()

    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)

    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()

    print('[Start] : pix2pix Training')
    for epoch in range(args.epochs):
        for step, data in enumerate(data_loader):
            real_A = to_variable(data['A'])
            real_B = to_variable(data['B'])
            fake_B = G(real_A)

            # Train Discriminator
            D_fake = D(torch.cat((real_A, fake_B), 1))
            D_real = D(torch.cat((real_A, real_B), 1))

            D_loss = 0.5 * patch_loss(criterion, D_fake, False) + 0.5 * patch_loss(criterion, D_real, True)

            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()

            # Train Generator

            D_fake = D(torch.cat((real_A, fake_B), 1))

            G_loss = patch_loss(criterion, D_fake, True) + euclidean_l1(fake_B, real_B) * args.lamda

            G_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            G_optimizer.step()

            if (step + 1) % 100 == 0:
                print("Epoch[{epoch}] |  Step [{now}/{total}] : D Loss : {D_loss}, G_losss : {G_loss}".format(epoch=epoch, now=step + 1, total=len(data_loader), D_loss=D_loss.item(), G_loss=G_loss.item()))
               
                #check 
                batch_image = (torch.cat((torch.cat((real_A, fake_B), 3), real_B), 3))
                torchvision.utils.save_image(denorm(batch_image[0]), './training_result/' + 'result_ep{epoch}_{step}.jpg'.format(epoch=epoch, step=(step + 1) * 4))
        
        torch.save(G.state_dict(), args.save_model + 'model_pix2pix_ep{epoch}'.format(epoch=epoch))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch pix2pix implementation')

    parser.add_argument('--dataset', default='./data/datasets/edges2handbags/', type=str, help='dataset path')
    parser.add_argument('--save_model', default='./data/', type=str, help='model save directory')

    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--betas', default=(0.5, 0.999), type=tuple, help='betas default : (0.5, 0.999)')
    parser.add_argument('--lamda', default=100, type=int, help='lamda')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')

    args = parser.parse_args()

    main()