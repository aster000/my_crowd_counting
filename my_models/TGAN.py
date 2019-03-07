from torchvision import models
import torch
import math
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None


    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


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

class Simple_D(nn.Module):
    def __init__(self):
        super(Simple_D, self).__init__()
        self.spp = SPPLayer(4)


       

        self.fc = nn.Sequential(
            nn.Linear(30*16, 100),
            nn.ReLU(),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self, feature_map, constant) :
        z_fc = self.spp(feature_map)
        z_fc = GradReverse.grad_reverse(z_fc, constant)
        domain_out = self.fc(z_fc)
        return domain_out
        
class TGAN(nn.Module):
    def __init__(self):
        super(TGAN, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=False)
        a = list(vgg16.children())[0]
        seq = list(a)[:14]
        self.vgg16 = nn.Sequential(*seq)
        
        self.D = Simple_D()


        self.feature_layer = nn.Sequential(
            nn.Conv2d(128, 64, (3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3,3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.density_layer = nn.Sequential(
            
            nn.Conv2d(16, 1, (1,1))
        )

    def forward(self, x, constant):
        y = self.vgg16(x)
        feature_map = self.feature_layer(y)
        density_map = self.density_layer(feature_map)
        domain_out  =  self.D(feature_map, constant)
        return density_map, domain_out

