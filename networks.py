import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


# generator+resnet block arch from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super(Generator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(64),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(64 * mult,
                                64 * mult * 2,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=use_bias),
                      norm_layer(64 * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(64 * mult,
                                  padding_type='reflect',
                                  norm_layer=norm_layer,
                                  use_dropout=True,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(64 * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                 bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True

        model = [
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 256, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, stride=2, padding=1, bias=use_bias),
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
