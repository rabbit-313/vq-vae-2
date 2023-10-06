import argparse
import os
from os import path as osp
import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from tqdm import tqdm
from vqvae import VQVAE
from torchvision.utils import save_image


def main(args):
    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # transform_ex = transforms.Compose(
    #     [
    #         transforms.Resize(256),
    #         transforms.CenterCrop(256),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    dataset = datasets.ImageFolder(args.dataset, transform=transform)
    loader = DataLoader(dataset)
    pbar = tqdm(loader)
          
    model = VQVAE()
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)
    model.eval()
    i=0
    for img, _ in pbar:
        img = img.to(device)
        i += 1
        with torch.no_grad():
            out, _ = model(img)
            
        os.makedirs(args.savefolder, exist_ok=True)
        save_image(out, osp.join(args.savefolder, f'f_{i:05d}.png'))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--savefolder", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
