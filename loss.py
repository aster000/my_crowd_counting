import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F



class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.sum((pred - target)**2) / pred.size(0)
        return loss

class SIMSE(nn.Module):
    
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, target):
        loss = torch.sum(target-pred).pow(2) / (pred.size(0) ** 2)
        return loss

class DiffLoss(nn.Module):
    
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)


        diff_loss = torch.mean((input1_l2.t().mm(input2_l2).pow(2)))

        return diff_loss
