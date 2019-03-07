from torchvision import models
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


        
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=False)
        a = list(vgg16.children())[0]
        seq = list(a)[:16]
        self.vgg16 = nn.Sequential(*seq)
        


        self.density_layer = nn.Sequential(
            nn.Conv2d(256, 128, (3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, (3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, (1,1)),
        )

    def forward(self, x):
        y = self.vgg16(x)
        density_map = self.density_layer(y)
        return density_map

class Baseline_DSN(nn.Module):
    def __init__(self):
        super(Baseline_DSN, self).__init__()

        self.encoder = Encoder()
        self.fc = nn.Sequential(
            FC(),
            FC2()
        )
        self.density_map = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,1,1)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        y = y.view(-1, 30*40*64)
        z = self.fc(y)
        z = z.view(-1, 64, 15, 20)
        density_map = self.density_map(z)
        return density_map

        z = self.fc(y)


class Baseline_us(nn.Module):
    def __init__(self):
        super(Baseline_us, self).__init__()

        self.encoder = encoder2()
        self.density_map = decoder2('density_map')
    def forward(self, x):
        y = self.encoder(x)
        z = self.density_map(y)
        return z
