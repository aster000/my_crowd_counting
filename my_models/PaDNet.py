from torchvision import models
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch
import math



class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() 
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        def weightd_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.02)
        self.spp = SPPLayer(4)
        self.fc = nn.Sequential(
                    nn.Linear(30*3, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),
                    nn.ReLU())
        self.fc.apply(weightd_init)
        
    def forward(self, x):
        y = self.spp(x)
        out = self.fc(y)
        return out

class PaDNet_Attention(nn.Module):
    def __init__(self):
        super(PaDNet_Attention, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.feature_extract = nn.Sequential(*seq)
        self.Attention = Attention()

        
        self.l3 = nn.Sequential()
        self.l3.add_module('my_conv_l3_1', nn.Conv2d(512, 128, (5,5), padding=(2,2)))
        self.l3.add_module('my_bn_l3_1', nn.BatchNorm2d(128))
        self.l3.add_module('my_relu_l3_1', nn.ReLU())
        self.l3.add_module('my_conv_l3_2', nn.Conv2d(128, 64, (5,5), padding=(2,2)))
        self.l3.add_module('my_bn_l3_2', nn.BatchNorm2d(64))
        self.l3.add_module('my_relu_l3_2', nn.ReLU())
        self.l3.add_module('my_conv_l3_3', nn.Conv2d(64, 32, (3,3), padding=(1,1)))
        self.l3.add_module('my_bn_l3_3', nn.BatchNorm2d(32))
        self.l3.add_module('my_relu_l3_3', nn.ReLU())
        self.l3.add_module('my_conv_l3_4', nn.Conv2d(32, 16, (3,3), padding=(1,1)))
        self.l3.add_module('my_bn_l3_4', nn.BatchNorm2d(16))
        self.l3.add_module('my_relu_l3_4', nn.ReLU())
        self.l3.add_module('my_conv_l3_5', nn.Conv2d(16, 1, (1,1)))


        self.l2 = nn.Sequential()
        self.l2.add_module('my_conv_l2_1', nn.Conv2d(512, 256, 7, padding=3))
        self.l2.add_module('my_bn_l2_1', nn.BatchNorm2d(256))
        self.l2.add_module('my_relu_l2_1', nn.ReLU())
        self.l2.add_module('my_conv_l2_2', nn.Conv2d(256, 128, 7, padding=3))
        self.l2.add_module('my_bn_l2_2', nn.BatchNorm2d(128))
        self.l2.add_module('my_relu_l2_2', nn.ReLU())
        self.l2.add_module('my_conv_l2_3', nn.Conv2d(128, 64, 5, padding=2))
        self.l2.add_module('my_bn_l2_3', nn.BatchNorm2d(64))
        self.l2.add_module('my_relu_l2_3', nn.ReLU())
        self.l2.add_module('my_conv_l2_4', nn.Conv2d(64, 32, 3, padding=1))
        self.l2.add_module('my_bn_l2_4', nn.BatchNorm2d(32))
        self.l2.add_module('my_relu_l2_4', nn.ReLU())
        self.l2.add_module('my_conv_l2_5', nn.Conv2d(32, 1, 1))


        self.l1 = nn.Sequential()
        self.l1.add_module('my_conv_l1_1', nn.Conv2d(512, 384, 9, padding=4))
        self.l1.add_module('my_bn_l1_1', nn.BatchNorm2d(384))
        self.l1.add_module('my_relu_l1_1', nn.ReLU())
        self.l1.add_module('my_conv_l1_2', nn.Conv2d(384, 256, 9, padding=4))
        self.l1.add_module('my_bn_l1_2', nn.BatchNorm2d(256))
        self.l1.add_module('my_relu_l1_2', nn.ReLU())
        self.l1.add_module('my_conv_l1_3', nn.Conv2d(256, 128, 7, padding=3))
        self.l1.add_module('my_bn_l1_3', nn.BatchNorm2d(128))
        self.l1.add_module('my_relu_l1_3', nn.ReLU())
        self.l1.add_module('my_conv_l1_4', nn.Conv2d(128, 64, 5, padding=2))
        self.l1.add_module('my_bn_l1_4', nn.BatchNorm2d(64))
        self.l1.add_module('my_relu_l1_4', nn.ReLU())
        self.l1.add_module('my_conv_l1_5', nn.Conv2d(64, 1, 1))


        self.fusion = nn.Sequential()
        self.fusion.add_module('my_conv_fusion_1', nn.Conv2d(6, 64, 7, padding=3))
        self.fusion.add_module('my_bn_fusion_1', nn.BatchNorm2d(64))
        self.fusion.add_module('my_relu_fusion_1', nn.ReLU())
        self.fusion.add_module('my_conv_fusion_2', nn.Conv2d(64, 32, 5, padding=2))
        self.fusion.add_module('my_bn_fusion_2', nn.BatchNorm2d(32))
        self.fusion.add_module('my_relu_fusion_2', nn.ReLU())
        self.fusion.add_module('my_conv_fusion_3', nn.Conv2d(32, 16, 3, padding=1))
        self.fusion.add_module('my_bn_fusion_3', nn.BatchNorm2d(16))
        self.fusion.add_module('my_relu_fusion_3', nn.ReLU())
        self.fusion.add_module('my_conv_fusion_4', nn.Conv2d(16, 1, 1))


    def forward(self, x, mode=None):
        feature = self.feature_extract(x)
        if mode == 'l1' :
            out = self.l1(feature)
        elif mode == 'l2':
            out = self.l2(feature)
        elif mode == 'l3':
            out = self.l3(feature)
        else:
            y1 = self.l1(feature)
            y2 = self.l2(feature)
            y3 = self.l3(feature)
            y_cat = torch.cat((y1, y2, y3), dim=1)
            fc = self.Attention(y_cat)
            fc = torch.transpose(fc, 1,0)
            fc_1, fc_2, fc_3 = torch.split(fc, 1)
            fc_1 = torch.transpose(fc_1, 1, 0).contiguous()
            fc_2 = torch.transpose(fc_2, 1, 0).contiguous()
            fc_3 = torch.transpose(fc_3, 1, 0).contiguous()
            bs, ch, w, d =y1.size()
            fc_1 = fc_1.view(bs, 1, 1, 1)
            fc_1 = fc_1.repeat(1, ch, w, d)
            fc_2 = fc_2.view(bs, 1, 1, 1)
            fc_2 = fc_2.repeat(1, ch, w, d)
            fc_3 = fc_3.view(bs, 1, 1, 1)
            fc_3 = fc_3.repeat(1, ch, w, d)

            y4 = torch.cat((y1, y2, y3, fc_1, fc_2, fc_3), dim=1)
            out = self.fusion(y4)
            
        return out

class PaDNet(nn.Module):
    def __init__(self):
        super(PaDNet, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.feature_extract = nn.Sequential(*seq)


        self.l3 = nn.Sequential()
        self.l3.add_module('my_conv_l3_1', nn.Conv2d(512, 128, (5,5), padding=(2,2)))
        self.l3.add_module('my_bn_l3_1', nn.BatchNorm2d(128))
        self.l3.add_module('my_relu_l3_1', nn.ReLU())
        self.l3.add_module('my_conv_l3_2', nn.Conv2d(128, 64, (5,5), padding=(2,2)))
        self.l3.add_module('my_bn_l3_2', nn.BatchNorm2d(64))
        self.l3.add_module('my_relu_l3_2', nn.ReLU())
        self.l3.add_module('my_conv_l3_3', nn.Conv2d(64, 32, (3,3), padding=(1,1)))
        self.l3.add_module('my_bn_l3_3', nn.BatchNorm2d(32))
        self.l3.add_module('my_relu_l3_3', nn.ReLU())
        self.l3.add_module('my_conv_l3_4', nn.Conv2d(32, 16, (3,3), padding=(1,1)))
        self.l3.add_module('my_bn_l3_4', nn.BatchNorm2d(16))
        self.l3.add_module('my_relu_l3_4', nn.ReLU())
        self.l3.add_module('my_conv_l3_5', nn.Conv2d(16, 1, (1,1)))


        self.l2 = nn.Sequential()
        self.l2.add_module('my_conv_l2_1', nn.Conv2d(512, 256, 7, padding=3))
        self.l2.add_module('my_bn_l2_1', nn.BatchNorm2d(256))
        self.l2.add_module('my_relu_l2_1', nn.ReLU())
        self.l2.add_module('my_conv_l2_2', nn.Conv2d(256, 128, 7, padding=3))
        self.l2.add_module('my_bn_l2_2', nn.BatchNorm2d(128))
        self.l2.add_module('my_relu_l2_2', nn.ReLU())
        self.l2.add_module('my_conv_l2_3', nn.Conv2d(128, 64, 5, padding=2))
        self.l2.add_module('my_bn_l2_3', nn.BatchNorm2d(64))
        self.l2.add_module('my_relu_l2_3', nn.ReLU())
        self.l2.add_module('my_conv_l2_4', nn.Conv2d(64, 32, 3, padding=1))
        self.l2.add_module('my_bn_l2_4', nn.BatchNorm2d(32))
        self.l2.add_module('my_relu_l2_4', nn.ReLU())
        self.l2.add_module('my_conv_l2_5', nn.Conv2d(32, 1, 1))


        self.l1 = nn.Sequential()
        self.l1.add_module('my_conv_l1_1', nn.Conv2d(512, 384, 9, padding=4))
        self.l1.add_module('my_bn_l1_1', nn.BatchNorm2d(384))
        self.l1.add_module('my_relu_l1_1', nn.ReLU())
        self.l1.add_module('my_conv_l1_2', nn.Conv2d(384, 256, 9, padding=4))
        self.l1.add_module('my_bn_l1_2', nn.BatchNorm2d(256))
        self.l1.add_module('my_relu_l1_2', nn.ReLU())
        self.l1.add_module('my_conv_l1_3', nn.Conv2d(256, 128, 7, padding=3))
        self.l1.add_module('my_bn_l1_3', nn.BatchNorm2d(128))
        self.l1.add_module('my_relu_l1_3', nn.ReLU())
        self.l1.add_module('my_conv_l1_4', nn.Conv2d(128, 64, 5, padding=2))
        self.l1.add_module('my_bn_l1_4', nn.BatchNorm2d(64))
        self.l1.add_module('my_relu_l1_4', nn.ReLU())
        self.l1.add_module('my_conv_l1_5', nn.Conv2d(64, 1, 1))


        self.fusion = nn.Sequential()
        self.fusion.add_module('my_conv_fusion_1', nn.Conv2d(3, 64, 7, padding=3))
        self.fusion.add_module('my_bn_fusion_1', nn.BatchNorm2d(64))
        self.fusion.add_module('my_relu_fusion_1', nn.ReLU())
        self.fusion.add_module('my_conv_fusion_2', nn.Conv2d(64, 32, 5, padding=2))
        self.fusion.add_module('my_bn_fusion_2', nn.BatchNorm2d(32))
        self.fusion.add_module('my_relu_fusion_2', nn.ReLU())
        self.fusion.add_module('my_conv_fusion_3', nn.Conv2d(32, 16, 3, padding=1))
        self.fusion.add_module('my_bn_fusion_3', nn.BatchNorm2d(16))
        self.fusion.add_module('my_relu_fusion_3', nn.ReLU())
        self.fusion.add_module('my_conv_fusion_4', nn.Conv2d(16, 1, 1))


    def forward(self, x, mode=None):
        feature = self.feature_extract(x)
        if mode == 'l1' :
            out = self.l1(feature)
        elif mode == 'l2':
            out = self.l2(feature)
        elif mode == 'l3':
            out = self.l3(feature)
        else:
            y1 = self.l1(feature)
            y2 = self.l2(feature)
            y3 = self.l3(feature)
            y_cat = torch.cat((y1, y2, y3), dim=1)
            out = self.fusion(y_cat)
        return out
