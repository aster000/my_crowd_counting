import torch
import os, gc
import torch.utils.data as data
from torchvision import transforms, datasets


data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])]
)

class Mnist(data.Dataset):
    
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test']


        self.mode = mode
        self.args = args

        if mode == 'train':
            self.load_train_data()
            self.image_num = 55000
        else:
            self.load_test_data()
            self.image_num = 10000
        
        self.hash = {}
        for i, data in enumerate(self.dataset):
            self.hash[i] = data
    def load_train_data(self):
        self.dataset = datasets.MNIST(root='/mnt/pami/yktian/dataset/', train=True, transform=data_tf, download=True) 

    def load_test_data(self):
        self.dataset = datasets.MNIST(root='/mnt/pami/yktian/dataset/', train=False, transform = data_tf)


    def __len__(self):
        return self.image_num

    def __getitem__(self, index):
        out = [index, self.hash[index]]
        return out
