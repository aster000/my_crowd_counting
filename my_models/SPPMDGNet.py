from torchvision import models
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
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


class SPPMDGNet(nn.Module):
    def __init__(self):
        super(SPPMDGNet, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        a = list(vgg16.children())[0]
        seq = list(a)[:23]
        self.vgg16_finetune = nn.Sequential(*seq)

        '''
        self.spp_low = SPPLayer(3)
        self.spp_mid = SPPLayer(3)
        self.spp_high = SPPLayer(3)
        '''
        self.spp = SPPLayer(3)
        self.high = nn.Sequential()
        #self.high.add_module('my_conv1', nn.Conv2d(512, 512, (3,3), padding=(2,2), dilation=2))
        #self.high.add_module('my_bn1', nn.BatchNorm2d())
        #self.high.add_module('my_relu1', nn.ReLU())
        self.high.add_module('my_conv_high1', nn.Conv2d(512, 128, (5,5), padding=(2,2), dilation=1))
        self.high.add_module('my_bn_high1', nn.BatchNorm2d(128))
        self.high.add_module('my_relu_high1', nn.ReLU())
        self.high.add_module('my_conv_high2', nn.Conv2d(128, 64, (5,5), padding=(2,2), dilation=1))
        self.high.add_module('my_bn_high2', nn.BatchNorm2d(64))
        self.high.add_module('my_relu_high2', nn.ReLU())
        self.high.add_module('my_conv_high3', nn.Conv2d(64, 32, (3,3), padding=(1,1), dilation=1))
        self.high.add_module('my_bn_high3', nn.BatchNorm2d(32))
        self.high.add_module('my_relu_high3', nn.ReLU())
        self.high.add_module('my_conv_high4', nn.Conv2d(32, 16, (3,3), padding=(1,1), dilation=1))
        self.high.add_module('my_bn_high4', nn.BatchNorm2d(16))
        self.high.add_module('my_relu_high4', nn.ReLU())
        #self.high.add_module('my_conv6', nn.Conv2d(128, 64, (3,3), padding=(2,2), dilation=2))
        #self.high.add_module('my_bn6', nn.BatchNorm2d(64))
        #self.high.add_module('my_relu6', nn.ReLU())
        self.high.add_module('my_conv_high5', nn.Conv2d(16,  1, (1,1)))
        #self.high.add_module('my_bn7', nn.BatchNorm2d(1))
        #self.high.add_module('my_relu7', nn.ReLU())

        self.mid = nn.Sequential()
        #self.mid.add_module('my_conv11', nn.Conv2d(512, 512, (5,5), padding=(4,4), dilation=2))
        #self.high.add_module('my_bn11', nn.BatchNorm2d())
        #self.mid.add_module('my_relu11', nn.ReLU())
        self.mid.add_module('my_conv_mid1', nn.Conv2d(512, 256, (7,7), padding=(3,3), dilation=1))
        self.mid.add_module('my_bn_mid1', nn.BatchNorm2d(256))
        self.mid.add_module('my_relu_mid1', nn.ReLU())
        self.mid.add_module('my_conv_mid2', nn.Conv2d(256, 128, (7,7), padding=(3,3), dilation=1))
        self.mid.add_module('my_bn_mid2', nn.BatchNorm2d(128))
        self.mid.add_module('my_relu_mid2', nn.ReLU())
        self.mid.add_module('my_conv_mid3', nn.Conv2d(128, 64, (5,5), padding=(2,2), dilation=1))
        self.mid.add_module('my_bn_mid3', nn.BatchNorm2d(64))
        self.mid.add_module('my_relu_mid3', nn.ReLU())
        self.mid.add_module('my_conv_mid4', nn.Conv2d(64, 32, (3,3), padding=(1,1), dilation=1))
        self.mid.add_module('my_bn_mid4', nn.BatchNorm2d(32))
        self.mid.add_module('my_relu_mid4', nn.ReLU())
        #self.mid.add_module('my_conv61', nn.Conv2d(128, 64, (5,5), padding=(4,4), dilation=2))
        #self.mid.add_module('my_bn61', nn.BatchNorm2d(64))
        #self.mid.add_module('my_relu61', nn.ReLU())
        self.mid.add_module('my_conv_mid5', nn.Conv2d(32,  1, (1,1)))
        #self.mid.add_module('my_bn71', nn.BatchNorm2d(1))
        #self.mid.add_module('my_relu71', nn.ReLU())

        self.low = nn.Sequential()
        #self.low.add_module('my_conv12', nn.Conv2d(512, 512, (7,7), padding=(6,6), dilation=2))
        #self.high.add_module('my_bn12', nn.BatchNorm2d())
        #self.low.add_module('my_relu12', nn.ReLU())
        self.low.add_module('my_conv_low1', nn.Conv2d(512, 384, (9,9), padding=(4,4), dilation=1))
        self.low.add_module('my_bn_low1', nn.BatchNorm2d(384))
        self.low.add_module('my_relu_low1', nn.ReLU())
        self.low.add_module('my_conv_low2', nn.Conv2d(384, 256, (9,9), padding=(4,4), dilation=1))
        self.low.add_module('my_bn_low2', nn.BatchNorm2d(256))
        self.low.add_module('my_relu_low2', nn.ReLU())
        self.low.add_module('my_conv_low3', nn.Conv2d(256, 128, (7,7), padding=(3,3), dilation=1))
        self.low.add_module('my_bn_low3', nn.BatchNorm2d(128))
        self.low.add_module('my_relu_low3', nn.ReLU())
        self.low.add_module('my_conv_low4', nn.Conv2d(128, 64, (5,5), padding=(2,2), dilation=1))
        self.low.add_module('my_bn_low4', nn.BatchNorm2d(64))
        self.low.add_module('my_relu_low4', nn.ReLU())
        #self.low.add_module('my_conv62', nn.Conv2d(128, 64, (7,7), padding=(6,6), dilation=2))
        #self.low.add_module('my_bn62', nn.BatchNorm2d(64))
        #self.low.add_module('my_relu62', nn.ReLU())
        self.low.add_module('my_conv_low5', nn.Conv2d(64,  1, (1,1)))
        #self.low.add_module('my_bn72', nn.BatchNorm2d(1))
        #self.low.add_module('my_relu72', nn.ReLU())
        '''
        self.fc_low = nn.Sequential()
        self.fc_low.add_module('my_conv_fc1', nn.Linear(14, 1))

        self.fc_mid = nn.Sequential()
        self.fc_mid.add_module('my_conv_fc2', nn.Linear(14, 1))

        self.fc_high = nn.Sequential()
        self.fc_high.add_module('my_conv_fc3', nn.Linear(14, 1))
        '''
        self.fc = nn.Sequential()
        self.fc.add_module('my_conv_fc', nn.Linear(42, 3))
        self.output =  nn.Sequential()
        #self.output.add_module('my_conv32432', nn.Conv2d(3, 1, (1,1))
        self.output.add_module('my_conv_final_1', nn.Conv2d(3, 64, (7,7), padding = (3,3)))
        self.output.add_module('my_bn_final_1', nn.BatchNorm2d(64))
        self.output.add_module('my_relu_final_1', nn.ReLU())
        self.output.add_module('my_conv_final_2', nn.Conv2d(64, 32, (5, 5), padding = (2,2)))
        self.output.add_module('my_bn_final_2', nn.BatchNorm2d(32))
        self.output.add_module('my_relu_final_2', nn.ReLU())
        self.output.add_module('my_conv_final_3', nn.Conv2d(32, 32, (3,3), padding = (1,1)))
        self.output.add_module('my_bn_final_3', nn.BatchNorm2d(32))
        self.output.add_module('my_relu_final_3', nn.ReLU())
        self.last = nn.Sequential()
        self.last.add_module('my_conv_last1', nn.Conv2d(32, 1, (1,1)))
        #self.output.add_module('my_conv_final_merge', nn.Conv2d(32, 1, (1,1)))

    def forward(self, x, mode=None):
        assert mode in ['low', 'mid', 'high', 'total']
        y = self.vgg16_finetune(x)
        if mode == 'low':
            out = self.low(y)
        elif mode =='mid' :
            out = self.mid(y)
        elif mode =='high':
            out = self.high(y)
        else:
            y1 = self.low(y)
            y2 = self.mid(y)
            y3 = self.high(y)
            y_cat = torch.cat((y1,y2,y3), dim=1)

            '''
            y_spp = self.spp(y_cat)

            y_fc = self.fc(y_spp)    
            

            y_fc_soft = F.softmax(y_fc, dim=1) + 1.0

            y_fc_soft = torch.transpose(y_fc_soft, 1,0)
            y1_2, y2_2, y3_2 = torch.split(y_fc_soft, 1)

            y1_2 = torch.transpose(y1_2, 1,0).contiguous()
            y2_2 = torch.transpose(y2_2, 1,0).contiguous()
            y3_2 = torch.transpose(y3_2, 1,0).contiguous()

            bs, ch, w, d = y1.size()
            y1_2 = y1_2.view(bs, 1, 1, 1)
            y1_2 = y1_2.repeat(1, ch, w, d)
            y1_3 = y1.mul(y1_2)

            y2_2 = y2_2.view(bs, 1, 1, 1)
            y2_2 = y2_2.repeat(1, ch, w, d)
            y2_3 = y2.mul(y2_2)

            y3_2 = y3_2.view(bs, 1, 1, 1)
            y3_2 = y3_2.repeat(1, ch, w, d)
            y3_3 = y3.mul(y3_2)

            z = torch.cat((y1_3, y2_3, y3_3), dim=1)
            y4 = self.output(z)
            '''
            y4 = self.output(y_cat)
            #y5 = torch.cat((y4, y1, y2, y3), dim=1)
            out = self.last(y4)
        if mode == 'total':
            #return out, y_fc
            return out
        else:
            return out
