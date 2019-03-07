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



class SSNet(nn.Module):
    def __init__(self):
        super(SSNet, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        
        self.en64 = nn.Sequential(*seq[0:4])
        self.en128 = nn.Sequential(*seq[4:9])
        self.en256 = nn.Sequential(*seq[9:16])
        self.en512 = nn.Sequential(*seq[16:23])
        self.de256 = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )



        self.de128 = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128,3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)

        )
        self.de64 = nn.Sequential(
            nn.Conv2d(256, 128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        
        )

        self.non_encoder = encoder2()

        self.density_map = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, mode='superviced', train=True):
        if train == True:
            en64feature = self.en64(x)
            en128feature = self.en128(en64feature)
            en256feature = self.en256(en128feature)
            en512feature = self.en512(en256feature)

            non_feature = self.non_encoder(x)

            total_feature = non_feature + en512feature

            de256feature = self.de256(total_feature)
            cat256feature = torch.cat((en256feature, de256feature), dim=1)
            de128feature = self.de128(cat256feature)
            cat128feature = torch.cat((en128feature, de128feature), dim=1)
            de64feature = self.de64(cat128feature)
            cat64feature = torch.cat((en64feature, de64feature), dim=1)
            recon = self.output(cat64feature)

            if mode == 'superviced':
                density_map = self.density_map(en512feature)
                return density_map, recon
            else:
                return recon
        else:
            en64feature = self.en64(x)
            en128feature = self.en128(en64feature)
            en256feature = self.en256(en128feature)
            en512feature = self.en512(en256feature)
            density_map = self.density_map(en512feature)
            return density_map
            
        
