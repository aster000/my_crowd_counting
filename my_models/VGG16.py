import torch
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False, num_classes=10)

    def forward(self, x):
        y = self.vgg16(x)
        return y
