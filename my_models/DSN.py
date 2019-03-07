from torchvision import models
import torch
import math
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None




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

class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16,32,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        
        )

    def forward(self, x):
        y = self.layer(x)
        return y



class DSN(nn.Module):
    def __init__(self, code_size=500) :
        super(DSN, self).__init__()
        self.code_size = code_size


        #self.spp = SPPLayer(4)

        #private source encoder


        self.source_encoder_conv = Encoder()

        self.source_encoder_fc = FC()

        #private target encoder
        self.target_encoder_conv = Encoder()


        self.target_encoder_fc = FC()


        #shared encoder

        self.shared_encoder_conv = Encoder()

        self.shared_encoder_fc = FC()

        #density map generate

        self.density_map_fc = FC2(code_size)

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
        #domain cls
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

        self.decoder_fc = FC2(code_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 48, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),

        )

    def forward(self,input_data, mode, rec_scheme, p=0.0) :
        result = []

        if mode == 'source' :
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 30*40*64)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target' :
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 30*40*64)
            private_code = self.target_encoder_fc(private_feat)


        result.append(private_code)

        #shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 30*40*64)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)


        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        
        result.append(domain_label)

        #if mode == 'source':
        density_feat = self.density_map_fc(shared_code)
        density_feat = density_feat.view(-1,64,15,20 )
        density_map = self.density_map(density_feat)
        result.append(density_map)
        
        if rec_scheme == 'share' :
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code


        rec_vec = self.decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 64,15, 20)

        rec_code = self.decoder_conv(rec_vec)

        result.append(rec_code)

        return result



        


