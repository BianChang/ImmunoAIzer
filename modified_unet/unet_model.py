""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class modified_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(modified_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(128, 256)
        self.down3 = Down1(1024, 2048)
        self.down4 = Down1(2048, 2048)
        self.up1 = Up(4096, 1024, bilinear)
        self.up2 = Up(2048, 512, bilinear)
        self.up3 = Up(640, 160, bilinear)
        self.up4 = Up(176, 44, bilinear)
        self.outc = OutConv(44, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}]