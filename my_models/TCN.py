import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import torch.nn.functional as F


class Chomp3d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp3d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:,:,:-self.chomp_size[0], :, :].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv3d(n_inputs, n_outputs, kernel_size, stride=stride,
                            padding=padding, dilation=dilation))

        self.chomp1 = Chomp3d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout3d(dropout)
        
        self.conv2 = weight_norm(nn.Conv3d(n_outputs, n_outputs, kernel_size, stride=stride,
                            padding=padding, dilation=dilation))
        self.chomp2 = Chomp3d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout3d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv3d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+ res)

class TCN(nn.Module):
    def __init__(self, input_size=3, num_channels=[16, 32, 48], kernel_size=(2, 3, 3), dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i ==0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=(dilation,1,1), stride=(1,1,1),
                                        padding=(((kernel_size[0]-1)*dilation),1,1), dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.output = nn.Sequential(nn.Conv3d(48, 1, kernel_size=1),nn.ReLU())

    def forward(self, x):
        y =  self.network(x)
        z = self.output(y)
        return z
