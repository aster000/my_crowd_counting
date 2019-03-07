import os, sys
import math, gc
import numpy as np
import torch
import pickle
import torch.utils.data as data
from torchvision import transforms



class Cifar10Dataset(data.Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test']

        self.mode = mode
        self.args = args
        if mode == 'train':
            self.image_num = args['data']['train']['number']
        else:
            self.image_num = args['data']['test']['number']


        gc.collect()

        if mode == 'train':
            self.load_train_data()
        else:
            self.load_test_data()

    def load_train_data(self):
        self.image_list = []
        self.label_list = []

        path = '/mnt/pami/yktian/dataset/cifar-10-batches-py/'
        for i in range(1,6):
            dataset = path + 'data_batch_' + str(i)
            with open(dataset, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                image = d[b'data']
                image = np.reshape(image, (10000, 3,32, 32))
                self.image_list = self.image_list + list(image)
                
                labels = d[b'labels']
                self.label_list = self.label_list + labels
        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[0,0,0], std=[255,255,255])
        for i in  range(len(self.image_list)):
            self.image_list[i] = normalizer(to_tensor(self.image_list[i]))

            
    def load_test_data(self):
        dataset = '/mnt/pami/yktian/dataset/cifar-10-batches-py/test_batch'
        with open(dataset, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            image = d[b'data']
            image = np.reshape(image, (10000, 3, 32, 32))
            self.image_list =  list(image)

            labels = d[b'labels']
            self.label_list = labels

        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[0,0,0], std=[255,255,255])
        for i in  range(len(self.image_list)):
            self.image_list[i] = normalizer(to_tensor(self.image_list[i]))

    def __len__(self):
        return self.image_num

    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]

        label = torch.Tensor([label])
        return [index, image, label]

class ToTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic).contiguous()
        return img.float()
