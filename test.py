import argparse
import sys
from os import path as osp

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
import matplotlib.pyplot as plt

import torchvision
from torchvision.utils import save_image
import glob
from PIL import Image
import PIL

def main(args):
    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    transform_ex = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(args.dataset, transform=transform)
    loader = DataLoader(dataset)
    pbar = tqdm(loader)
          
    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()
    i=0
    for img, _ in pbar:
        img = img.to(device)
        i += 1
        with torch.no_grad():
            out, _ = model(img)
        # 変更 step1
        
        
        # out = out.squeeze(0)
        # out = transforms.ToPILImage()(out)
        # plt.imshow(out)
        # plt.axis('off')
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.savefig(osp.join(args.save_folder, f'f_{i:05d}.svg'), bbox_inches = 'tight', pad_inches = 0)
        # # out.save(f'f_{i:05d}.svg')
        # # out.save(osp.join(args.save_folder, f'f_{i:05d}.png'))
        save_image(out, osp.join(args.save_folder, f'f_{i:05d}.png'))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("save_folder", type=str)
    args = parser.parse_args()
    print(args)
    main(args)