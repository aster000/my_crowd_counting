import os, sys
import math, gc
import PIL.Image
import numpy as np
import torch
import pickle
import torch.utils.data as data
from torchvision import transforms


class ShanghaiA(data.Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test']


        self.mode = mode
        self.args = args

        gc.collect()
        


        if mode == 'train':
            self.load_train_data()
        else:
            self.load_test_data()

    def load_train_data(self):
        def load(path):
            return PIL.Image.open(path).convert('RGB')
        self.dataset_pkl = open(self.args['dataset_path'], 'rb')
        self.density_pkl = open(self.args['density_map'], 'rb')
        dataset = pickle.load(self.dataset_pkl)
        data_density = pickle.load(self.density_pkl)
        self.image_name_list = dataset['image_name_list']
        self.image_pos = dataset['image_pos']
        self.density_list = data_density['train_list']
        self.density_flip_list = data_density['train_flip_list']

        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[0,0,0], std=[255,255,255])
        self.image_list = []
        self.data_length = len(self.image_name_list)
        for i in range(self.data_length):
            if i % 50 ==0:
                print(f'Initialize data....{i/self.data_length*100 : 0.2f}%\r', end='')
            self.image_list.append(normalizer(to_tensor(load(self.image_name_list[i]))))

        self.dataset_pkl.close()
        self.density_pkl.close()

    def load_test_data(self):
        def load(path):
            return PIL.Image.open(path).convert('RGB')


        self.dataset_pkl = open(self.args['dataset_path'], 'rb')
        self.density_pkl = open(self.args['density_map'], 'rb')

        dataset = pickle.load(self.dataset_pkl)
        data_density = pickle.load(self.density_pkl)
        self.image_name_list = dataset['test_name_list']
        self.density_list = data_density['test_list']

        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[0,0,0], std=[255,255,255])
        self.image_list = []
        self.data_length = len(self.image_name_list)
        for i in range(self.data_length):
            if i % 50 == 0 :
                print(f'Initialize data....{i/self.data_length*100  : 0.2f}%\r', end='')
            self.image_list.append(normalizer(to_tensor(load(self.image_name_list[i]))))

        self.dataset_pkl.close()
        self.density_pkl.close()

    def get_index(self, name):
        index = name.find('PATCH')
        index += 6
        s = ''
        while True:
            if name[index] <= '9' and name[index] >= '0' :
                s = s + name[index]
                index += 1
            else:
                break
        return int(s)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        image_name = self.image_name_list[index]

        if self.mode == 'train':
            h, w, h_len, w_len = self.image_pos[index].astype(np.int) // self.args['downscale']
            if 'flip' not in image_name:
                num = self.get_index(image_name)
                density = self.density_list[num-1]
            else :
                num = self.get_index(image_name)
                density = self.density_flip_list[num-1]
            density = torch.from_numpy(density[h:h+h_len, w:w+w_len])
        else:
            density = self.density_list[index]
            density = torch.from_numpy(density)
        density = density.unsqueeze(0)

        return [index, image, density]
    def __len__(self):
        return self.data_length


class ToTensor(object):
    def __call__(self, pic):
        if pic.mode == 'RGB':
            nchannel = 3
            img = np.asarray(pic)
        elif pic.mode == 'L':
            img = np.asarray(pic)[:,:,np.newaxis]
        else:
            raise Exception('Undefined mode: {}'.format(pic.mode))

        img = torch.from_numpy(img).transpose(0,1).transpose(0,2).contiguous()
        return img.float()
