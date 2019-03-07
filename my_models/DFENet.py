from torchvision import models
import torch
import math
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable


class encoder1(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.layer = nn.Sequential(*seq)


    def forward(self, x):
        y = self.layer(x)
        return y

class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()

        vgg16 = models.vgg16_bn(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:33]
        self.layer = nn.Sequential(*seq)


    def forward(self, x):
        y = self.layer(x)
        return y

class decoder1(nn.Module):
    def __init__(self, mode='decoder'):
        super(decoder1, self).__init__()

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
                nn.Conv2d(64, 3, 3, padding=1)
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(64, 1, 1)
            )
         
    def forward(self, x):
        y = self.layer(x)
        z = self.output(y)
        return z


class decoder2(nn.Module):
    def __init__(self, mode='decoder'):
        super(decoder2, self).__init__()

        self.layer  = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
            )
         
    def forward(self, x):
        y = self.layer(x)
        return y


class FC(nn.Module):
    def __init__(self, fc_size=500):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
           nn.Linear(512*45*45, fc_size),
           nn.ReLU()
        )

    def forward(self,x):
        y = self.fc(x)
        return y


class FC2(nn.Module):
    def __init__(self, fc_size=500):
        super(FC2, self).__init__()
        self.fc = nn.Sequential(
           nn.Linear(fc_size, 512*45*45),
           nn.ReLU()
        )

    def forward(self,x):
        y = self.fc(x)
        return y

class DFENet(nn.Module):
    def __init__(self, fc_size=500):
        super(DFENet, self).__init__()

        self.non_encoder = encoder2()
        self.crowd_encoder = encoder2()
        self.recon = decoder1()
        self.density_map = decoder2(mode='density')


        
    def forward(self, x, train=True):
        non_feature = self.non_encoder(x)

        crowd_feature = self.crowd_encoder(x)


        total_feature = crowd_feature + non_feature 



        
        density_map = self.density_map(crowd_feature)
        recon_image = self.recon(total_feature)
        return density_map, recon_image, crowd_feature, non_feature
            
