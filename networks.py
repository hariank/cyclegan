import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, 64, 7, bias=True),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]

        # Add downsampling layers
        n_downsampling = 2
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(64 * mult,
                                64 * mult * 2,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True),
                      nn.InstanceNorm2d(64 * mult * 2),
                      nn.ReLU(True)]

        # Add resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(64 * mult)]

        # Add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(64 * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)  # skip connection


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 256, 4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, stride=2, padding=1, bias=True),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        batch_dim = x.shape[0]
        out = self.model(x)
        return F.avg_pool2d(out, out.shape[-1]).view(batch_dim, 1)


def init_weights_gaussian(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight, 0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0.)


def gen_gan_loss(fake_score):
    return F.mse_loss(fake_score, torch.ones_like(fake_score))


def disc_gan_loss(real_score, fake_score):
    return F.mse_loss(real_score, torch.ones_like(real_score)) + \
        F.mse_loss(fake_score, torch.zeros_like(fake_score))


def gen_cyc_loss(original, reconstructed):
    return F.l1_loss(original, reconstructed)
