import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from parameters import *


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # input is Z, going into a convolution
        self.ct0 = nn.ConvTranspose2d(nz, 32*ngf, 4, 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(32*ngf)
        self.rl0 = nn.ReLU(True)
        # state size. (32*ngf) x 4 x 4
        self.ct1 = nn.ConvTranspose2d(32*ngf, 16*ngf, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*ngf)
        self.rl1 = nn.ReLU(True)
        # state size. (16*ngf) x 8 x 8
        self.ct2 = nn.ConvTranspose2d(16*ngf, 8*ngf, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(8*ngf)
        self.rl2 = nn.ReLU(True)
        # state size. (8*ngf) x 16 x 16
        self.ct3 = nn.ConvTranspose2d( 8*ngf, 4*ngf, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*ngf)
        self.rl3 = nn.ReLU(True)
        # state size. (4*ngf) x 32 x 32
        self.ct4 = nn.ConvTranspose2d( 4*ngf, 2*ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(2*ngf)
        self.rl4 = nn.ReLU(True)
        # state size. (2*ngf) x 64 x 64
        self.ct5 = nn.ConvTranspose2d( 2*ngf, ngf, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.rl5 = nn.ReLU(True)
        # state size. (ngf) x 128 x 128
        self.ct6 = nn.ConvTranspose2d(   ngf, nc, 4, 2, 1, bias=False)
        self.th0 = nn.Tanh()
        # state size. (nc) x 256 x 256

    def forward(self, x):
        x = self.rl0(self.bn0(self.ct0(x)))
        x = self.rl1(self.bn1(self.ct1(x)))
        x = self.rl2(self.bn2(self.ct2(x)))
        x = self.rl3(self.bn3(self.ct3(x)))
        x = self.rl4(self.bn4(self.ct4(x)))
        x = self.rl5(self.bn5(self.ct5(x)))
        x = self.th0(self.ct6(x))

        return x


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*16) x 4 x 4
        )
        #self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.discriminate = nn.Sequential(
            nn.Linear(4*4*16*ndf, 1),
            nn.Sigmoid())
            #nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid())

        self.classify = nn.Sequential(
            #nn.Conv2d(16*ndf, n_class, 4, 1, 0, bias=False)
            nn.Linear(4*4*16*ndf, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, n_class)
            #nn.Softmax(dim=1)
            )

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0),-1)
        #x = self.pool(x)
        d_out = self.discriminate(x)
        c_out = self.classify(x)
        return d_out, c_out
