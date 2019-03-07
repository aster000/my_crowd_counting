from torchvision import models
import torch
import math
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        y = self.layer(x)
        return y

class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.layer = nn.Sequential(*seq)


    def forward(self, x):
        y = self.layer(x)
        return y

class encoder3(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.layer = nn.Sequential(*seq)


    def forward(self, x):
        y = self.layer(x)
        return y
class decoder2(nn.Module):
    def __init__(self, mode='decoder'):
        super(decoder2, self).__init__()

        self.layer  = nn.Sequential(
            nn.Conv2d(512, 512,3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128,3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(128, 64,3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU()
        )

        if mode == 'decoder':
            self.output = nn.Sequential(
                nn.Conv2d(64, 3, 3, 1)
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(64, 1, 1)
            )
         
    def forward(self, x):
        y = self.layer(x)
        z = self.output(y)
        return z


class decoder3(nn.Module):
    def __init__(self, mode='decoder'):
        super(decoder2, self).__init__()

        self.layer  = nn.Sequential(
            nn.Conv2d(512, 512,3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128,3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64,3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )

        if mode == 'decoder':
            self.output = nn.Sequential(
                nn.Conv2d(64, 3, 3, 1)
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(64, 1, 1)
            )
         
    def forward(self, x):
        y = self.layer(x)
        z = self.output(y)
        return z
class decoder(nn.Module):
    def __init__(self, mode='decoder'):
        super(decoder, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        if mode == 'decoder':
            self.output = nn.Sequential(
                nn.Conv2d(16, 3, kernel_size=3, padding=1)
            )
        else :
            self.output = nn.Sequential(
                nn.Conv2d(16,1, kernel_size=1)
            )
    def forward(self, x):
        y = self.layer(x)
        z = self.output(y)
        return z



class USNet(nn.Module):
    def __init__(self):
        super(USNet, self).__init__()

        self.crowd_encoder = encoder()
        self.non_encoder = encoder()
        self.decoder = decoder('decoder')
        self.density_map = decoder('density_map')

    def forward(self, x, mode='superviced'):
        crowd_feature = self.crowd_encoder(x)
        non_feature = self.non_encoder(x)
        total_feature = crowd_feature + non_feature
        recon = self.decoder(total_feature)
        if mode == 'superviced':
            density_map = self.density_map(crowd_feature)
            return density_map, recon
        else:
            return recon
        
