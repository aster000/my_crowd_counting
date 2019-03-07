from torchvision import models
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init




class UNet_CSR(nn.Module):
    def __init__(self):
        super(UNet_CSR, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.encoder = nn.Sequential(*seq)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)

        return z

    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        vgg16 = models.vgg16(pretrained=False)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.encoder = nn.Sequential(*seq)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)

        return z

class UNet1(nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()

        self.en64 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.en128 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128,3 ,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128, 3, padding=1),
            nn.ReLU(),
        )
        self.en256 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.en512 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.de256 = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        
        self.de128 = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
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


    def forward(self, x):
        en64feature = self.en64(x)
        en128feature = self.en128(en64feature)
        en256feature = self.en256(en128feature)
        en512feature = self.en512(en256feature)
        de256feature = self.de256(en512feature)
        cat256feature = torch.cat((en256feature, de256feature), dim=1)
        de128feature = self.de128(cat256feature)
        cat128feature = torch.cat((en128feature, de128feature), dim=1)
        de64feature = self.de64(cat128feature)
        cat64feature = torch.cat((en64feature, de64feature), dim=1)
        output = self.output(cat64feature)
        return output


class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

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


    def forward(self, x):
        en64feature = self.en64(x)
        en128feature = self.en128(en64feature)
        en256feature = self.en256(en128feature)
        en512feature = self.en512(en256feature)
        de256feature = self.de256(en512feature)
        cat256feature = torch.cat((en256feature, de256feature), dim=1)
        de128feature = self.de128(cat256feature)
        cat128feature = torch.cat((en128feature, de128feature), dim=1)
        de64feature = self.de64(cat128feature)
        cat64feature = torch.cat((en64feature, de64feature), dim=1)
        output = self.output(cat64feature)
        return output
