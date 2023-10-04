import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

from models import FilterLow


def train(epoch, loader, model, optimizer, scheduler, device, kernel_size, gaussian, sample_folder_name):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    pixel_loss = nn.L1Loss()
    # color_filter = FilterLow(recursions=recursions, stride=stride, kernel_size=kernel_size, padding=False,gaussian=gaussian)
    
    if gaussian == "False":
        gaussian = False
    elif gaussian == "True":
        gaussian = True
        
    color_filter = FilterLow(kernel_size=kernel_size, gaussian=gaussian, include_pad=False)
    
    if torch.cuda.is_available():
        pixel_loss = pixel_loss.cuda()
        color_filter = color_filter.cuda()  
    
    
    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img) # MSE loss
        
        # ここに色損失を追加する
        # 入力と出力にローパスフィルタをかけてそれらのMSEを取る
        color_loss = pixel_loss(color_filter(out), color_filter(img))
        
        
        latent_loss = latent_loss.mean() # 量子化损失
        loss = recon_loss + latent_loss_weight * latent_loss + color_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"colorloss: {color_loss.item():.3f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 300 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                # フォルダが存在しない場合に作成する
                folder_path = f"sample/{sample_folder_name}"
                os.makedirs(folder_path, exist_ok=True)
                
                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{sample_folder_name}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, args.kernel_size, args.gaussian, args.sample)

        if dist.is_primary():
            folder_path = f"checkpoint/{args.checkpoint}"
            os.makedirs(folder_path, exist_ok=True)
            torch.save(model.state_dict(), f"checkpoint/{args.checkpoint}/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--gaussian", type=str, default="False")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--sample", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
