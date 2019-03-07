from torchvision import models
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
'''
vgg16 = models.vgg16(pretrained=True)
a = list(vgg16.children())[0]
seq = list(a)[:23]
vgg16_finetune = nn.Sequential(*seq)
print(vgg16_finetune)
#features = nn.Sequential(*list(vgg16.children())[0][:10])
#print(features)

'''
class CSRNet2(nn.Module):
    def __init__(self):
        super(CSRNet2, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.vgg16_finetune = nn.Sequential(*seq)


        self.dilate_layer = nn.Sequential()
        self.dilate_layer.add_module('my_conv2', nn.Conv2d(512, 256, (3,3), padding=(1,1)))
        self.dilate_layer.add_module('my_relu2', nn.ReLU())
        self.dilate_layer.add_module('my_conv3', nn.Conv2d(256, 128, (3,3), padding=(1,1)))
        self.dilate_layer.add_module('my_relu5', nn.ReLU())
        self.dilate_layer.add_module('my_conv6', nn.Conv2d(128, 64, (3,3), padding=(1,1)))
        self.dilate_layer.add_module('my_relu6', nn.ReLU())
        self.dilate_layer.add_module('my_conv7', nn.Conv2d(64,  1, (1,1)))
        self.dilate_layer.add_module('my_relu7', nn.ReLU())
        
    def forward(self, x):
        y = self.vgg16_finetune(x)
        out = self.dilate_layer(y)
        return out
            

