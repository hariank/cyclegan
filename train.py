import argparse
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from datasets import get_loaders
from networks import (Discriminator,
                      Generator,
                      disc_gan_loss,
                      gen_cyc_loss,
                      gen_gan_loss,
                      init_weights_gaussian)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--data', dest='data_dir', type=str, default='../data/maps')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--identity-loss', dest='identity_loss', action='store_true')
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--log-dir', dest='log_dir', type=str, default='tb')
    parser.add_argument('--nes', dest='num_epochs_save', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--use-cpu', dest='use_cpu', action='store_true')
    parser.add_argument('--gpu-ids', dest='gpu_ids', type=str, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device('cuda' if not args.use_cpu else 'cpu')
    args.loader_kwargs = {'num_workers': 4,
                          'pin_memory': True} if not args.use_cpu else {}

    ckpt_dir = os.path.join(args.ckpt_dir, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    jitter_size = args.resize + 30  # random jitter from pix2pix
    tf = transforms.Compose([transforms.Resize(jitter_size, Image.ANTIALIAS),
                             transforms.RandomCrop((args.resize, args.resize)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor()])
    train_loader, test_loader = get_loaders(args, tf)

    G = Generator(in_channels=3, out_channels=3, n_blocks=9).to(device)  # A to B
    F = Generator(in_channels=3, out_channels=3, n_blocks=9).to(device)  # B to A
    D_A = Discriminator(in_channels=3).to(device)
    D_B = Discriminator(in_channels=3).to(device)
    nets = [G, F, D_A, D_B]
    for net in nets:
        net.apply(init_weights_gaussian)
        net.train()

    G_opt = optim.Adam(G.parameters(), lr=args.lr)
    F_opt = optim.Adam(F.parameters(), lr=args.lr)
    D_A_opt = optim.Adam(D_A.parameters(), lr=args.lr)
    D_B_opt = optim.Adam(D_B.parameters(), lr=args.lr)
    opts = [G_opt, F_opt, D_A_opt, D_B_opt]

    for epoch in range(args.epochs):
        with tqdm(train_loader,
                  unit_scale=args.batch_size,
                  dynamic_ncols=True,
                  desc='Epoch {}'.format(epoch)) as pbar:
            for batch_idx, (x_A, x_B) in enumerate(pbar):
                x_A, x_B = x_A.to(device), x_B.to(device)

                for opt in opts:
                    opt.zero_grad()

                # train generators
                fake_A, fake_B = F(x_B), G(x_A)
                gan_loss = gen_gan_loss(fake_A) + gen_gan_loss(fake_B)
                cyc_loss = gen_cyc_loss(F(fake_B), x_A) + gen_cyc_loss(G(fake_A), x_B)
                gen_loss = gan_loss + 10. * cyc_loss
                gen_loss.backward()
                G_opt.step()
                F_opt.step()

                # train discriminators
                fake_A, fake_B = fake_A.detach(), fake_B.detach()
                disc_loss = disc_gan_loss(D_A(x_A), D_A(fake_A)) + \
                    disc_gan_loss(D_B(x_B), D_B(fake_B))
                disc_loss.backward()
                D_A_opt.step()
                D_B_opt.step()

                pbar.set_postfix(gen_loss=gen_loss.detach().item(),
                                 disc_loss=disc_loss.detach().item())
