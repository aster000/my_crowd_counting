import sys, os
import numpy as np
import gc, time
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import utility
import my_models
import torch.nn as nn
import torch.nn.init as init
from loss import MSELoss, SIMSE, DiffLoss
import data_folders



class Engine(object):
    def __init__(self, args, checkpoint_dir):
        self.args = args
        self.chk_dir = checkpoint_dir
        self.density_criterion = MSELoss().cuda()

        init_data = getattr(self, args['model']['init_data'])
        init_model_optimizer = getattr(self, args['model']['init_model_optimizer'])

        init_data()
        init_model_optimizer()

        self.train_recorder_list = ['time'] + args['train_recorder_list']
        self.test_recorder_list = ['time'] + args['test_recorder_list']

        self.train_loss = pd.DataFrame(columns=args['train_recorder_list'])
        self.test_loss = pd.DataFrame(columns=args['test_recorder_list'])

        self.epoch, self.best_record, self.train_loss, self.test_loss = self.load_checkpoint()

    def init_dataloader(self):
        print(f'Initializing data.....\r', end='')
        self.train_loader = self.get_dataloader('dataset', 'train')
        self.test_loader = self.get_dataloader('dataset', 'test')

    def init_model_optimizer(self):

        
        with utility.Timer('Initialize '+self.args['model']['network']+' model and optimizer') as t:
            self.model = my_models.__dict__[self.args['model']['network']]()
             
            if self.args['model']['optimizer'] in ['Adam', 'SGD', 'RMSprop']:
                self.optimizer = torch.optim.__dict__[self.args['model']['optimizer']](
                                    self.model.parameters(),
                                    lr=self.args['model']['learning_rate'],
                                    weight_decay = self.args['model']['weight_decay'])
            self.model = torch.nn.DataParallel(self.model).cuda()
        
    def get_dataloader(self, name, mode):
        dataset = self.args['data'][mode]['dataset']
        with utility.Timer('Initializing ' + name + ' '+ dataset['name'] +' '  + mode) as t:
            folder = data_folders.__dict__[dataset['name']](dataset, mode=mode)
            loader = data.DataLoader(folder,
                        batch_size = self.args['data'][mode]['batch_size'],
                        shuffle = self.args['data'][mode]['shuffle'],
                        num_workers = self.args['data'][mode]['num_workers'])
            return loader

  



    def init_recorder(self, key_list=['time']):
        recorder = {}
        for key in key_list:
            recorder[key] = utility.AverageMeter()
        return recorder

    def load_checkpoint(self):
        epoch = 0
        best_record = 999999
        train_loss = self.train_loss
        test_loss = self.test_loss
        return epoch, best_record, train_loss, test_loss

    def load_parameters(self, mode):
        model_dict = self.model.state_dict()

        if mode == 'total':
            load_file = self.args['model']['test_path']
            checkpoint = torch.load(load_file)
            pretrained_dict = checkpoint['state_dict']
            self.model.load_state_dict(pretrained_dict)

        else:
            load_file = self.chk_dir + '/' + mode + '_best_checkpoint.tar'
            checkpoint = torch.load(load_file)
            pretrained_dict = checkpoint['state_dict']
            param_dict = {k : v for k, v in pretrained_dict.items() if mode in k}
            model_dict.update(param_dict)
            self.model.load_state_dict(model_dict)


    def train_PaDNet(self, mode = 'total') :
        self.model.train()
        self.epoch += 1
        self.current_time = time.time()

        train_loader = self.train_loader[mode]
        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(train_loader)

        for i, (idx, image, density) in enumerate(train_loader):
            print('Training ' + mode + ' sub-network....' + f'{i/num_iter*100:0.1f}% \r', end='')

            input_var = Variable(image.cuda()).type(torch.cuda.FloatTensor)
            target_var = Variable(density.cuda()).type(torch.cuda.FloatTensor)
            self.optimizer.zero_grad()

            pred_var = self.model(x=input_var, mode=mode)
            loss = self.density_criterion(pred_var, target_var)

            pred_var = pred_var.clamp(min=-10, max=15)


            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()

        if mode == 'l1':
            self.update_recorder(recorder, l1_density_loss=loss, pred_density=pred_var, target_density=target_var, mode=mode)
        elif mode == 'l2' :
            self.update_recorder(recorder, l2_density_loss=loss, pred_density=pred_var, target_density=target_var, mode=mode)
        elif mode == 'l3' :
            self.update_recorder(recorder, l3_density_loss=loss, pred_density=pred_var, target_density=target_var, mode=mode)
        elif mode == 'total' :
            self.update_recorder(recorder, total_density_loss=loss, pred_density=pred_var, target_density=target_var, mode=mode)


        self.update_loss(recorder, 'train')
        utility.print_info(recorder, epoch=self.epoch, mode=mode, preffix='Train ' + mode + ' sub-network ' )
            
            

    def validate_PaDNet(self, mode='total'):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)

        result_density = []
        for i, (idx, image, density) in enumerate(self.test_loader):
            print(i)
            print(f'Validating ' + mode + ' sub-network....' + f'{i/num_iter*100:.1f}%\r', end='')

            input_var = Variable(image.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            density_var = Variable(density.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)

            pred_density = self.model(x= input_var, mode=mode)

            loss = self.density_criterion(pred_density, density_var)

            for i in range(idx.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[i])

            if mode =='l1':
                self.update_recorder(recorder, l1_density_loss=loss, pred_density=pred_density, target_density=density_var, mode=mode)
            elif mode == 'l2':
                self.update_recorder(recorder, l2_density_loss=loss, pred_density=pred_density, target_density=density_var, mode= mode)
            elif mode == 'l3':
                self.update_recorder(recorder, l3_density_loss=loss, pred_density=pred_density, target_density=density_var, mode=mode)
            elif mode == 'total':
                self.update_recorder(recorder, total_density_loss=loss, pred_density=pred_density, target_density=density_var, mode=mode)
        self.update_loss(recorder, 'test')

        print('[***** Best ' + mode +' MAE: {best_record:.2f} *****]'.format(best_record = self.best_record[mode]))
        utility.print_info(recorder, mode=mode, preffix='*** Validation *** '+mode+' sub-ntework ')
        self.save_checkpoint(result_dict={(mode+'_density'):result_density}, recorder=recorder, mode=mode)

    def validate_PaDNet_Patch(self, mode='total'):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)

        result_density = []
        for i, (idx, image, density) in enumerate(self.test_loader):
            print(f'Validating ' + mode + 'sub-network....' + f'{i/num_iter*100:.1f}%\r', end='')

            input_var = Variable(image.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            density_var = Variable(density.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            input_list = []
            density_list = []
            input_list.append(input_var[:,:,0:360,0:360])
            input_list.append(input_var[:,:,0:360,360:])
            input_list.append(input_var[:,:,360:,0:360])
            input_list.append(input_var[:,:,360:,360:])
            density_list.append(density_var[:,:,0:45,0:45])
            density_list.append(density_var[:,:,0:45,45:])
            density_list.append(density_var[:,:,45:,0:45])
            density_list.append(density_var[:,:,45:,45:])

            pred_density = torch.zeros(1,1,90,90)
            pred_density = Variable(pred_density.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            result_list = []
            for i in range(len(input_list)):
                pred = self.model(input_list[i])
                result_list.append(pred)

            pred_density[:,:,0:45,0:45] = result_list[0]
            pred_density[:,:,0:45,45:] = result_list[1]
            pred_density[:,:,45:,0:45] = result_list[2]
            pred_density[:,:,45:,45:] = result_list[3]

            loss = self.density_criterion(pred_density, density_var)
            
            for i in range(idx.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[i])

            if mode =='l1':
                self.update_recorder(recorder, l1_density_loss=loss, pred_density=pred_density, target_density=density_var, mode=mode)
            elif mode == 'l2':
                self.update_recorder(recorder, l2_density_loss=loss, pred_density=pred_density, target_density=density_var, mode= mode)
            elif mode == 'l3':
                self.update_recorder(recorder, l3_density_loss=loss, pred_density=pred_density, target_density=density_var, mode=mode)
            elif mode == 'total':
                self.update_recorder(recorder, total_density_loss=loss, pred_density=pred_density, target_density=density_var, mode=mode)
        self.update_loss(recorder, 'test')

        print('[***** Best ' + mode +' MAE: {best_record:.2f} *****]'.format(best_record = self.best_record[mode]))
        utility.print_info(recorder, mode=mode, preffix='*** Validation *** '+mode+' sub-ntework ')
        self.save_checkpoint(result_dict={(mode+'_density'):result_density}, recorder=recorder, mode=mode)

    def train_tcn(self):
        self.model.train()
        self.epoch += 1
        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(self.train_loader)
        self.current_time = time.time()

        for i, (images, density) in enumerate(self.train_loader):
            print(f'Training...{i/num_iter*100:0.1f}% \r', end='')

            input_var = Variable(images.cuda()).type(torch.cuda.FloatTensor)
            target_var = Variable(density.cuda()).type(torch.cuda.FloatTensor)[:,:,0,:,:]
            self.optimizer.zero_grad()



            pred_var = self.model(input_var)[:,:,-1,:,:]
            
            pred_var = pred_var.clamp(min=-10, max=15)

            loss = self.density_criterion(pred_var, target_var)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()

            self.update_recorder(recorder, density_loss=loss, pred_density=pred_var, target_density=target_var)
        self.update_loss(recorder, 'train')
        utility.print_info(recorder, epoch=self.epoch, preffix='Train')

    def validate_tcn(self):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)

        result_density = []

        for i, (images, density) in enumerate(self.test_loader):
            print(f'Validating.... {i/num_iter*100:.1f}% \r', end='')

            input_var = Variable(images.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            target_var = Variable(density.cuda()).type(torch.cuda.FloatTensor)[:,:,0,:,:]

            pred_density = self.model(input_var)[:,:,-1,:,:]
            loss = self.density_criterion(pred_density, target_var)

            for i in range(images.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[i])
            self.update_recorder(recorder, density_loss=loss, pred_density=pred_density, target_density=target_var)
        self.update_loss(recorder, 'test')
        print('[*** Best MAE: {best_record:.2f} ***]'.format(best_record = self.best_record))
        utility.print_info(recorder, preffix='*** Validating *** ')
        self.save_checkpoint(result_dict={'density':result_density}, recorder=recorder)
            
    def train_crowd(self):
        self.model.train()
        self.epoch += 1
        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(self.train_loader)
        self.current_time = time.time()



        for i, (idx, image, density) in enumerate(self.train_loader):
            print(f'Training...{i/num_iter*100:0.1f}% \r', end='')


            input_var = Variable(image.cuda()).type(torch.cuda.FloatTensor)
            target_var = Variable(density.cuda()).type(torch.cuda.FloatTensor)
            self.optimizer.zero_grad()

            pred_var = self.model(input_var)
            pred_var = pred_var.clamp(min=-10, max=15)

            loss = self.density_criterion(pred_var, target_var)


            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()


            self.update_recorder(recorder, density_loss=loss, pred_density=pred_var, target_density=target_var)
        self.update_loss(recorder, 'train')
        utility.print_info(recorder, epoch=self.epoch, preffix='Train')
    def train_st(self):
        self.model.train()
        self.epoch += 1
        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(self.train_loader)
        self.current_time = time.time()
        self.first = True


        

        for i, (idx, image, density) in enumerate(self.train_loader):
            print(f'Training...{i/num_iter*100:0.1f}% \r', end='')


            input_var = Variable(image.cuda()).type(torch.cuda.FloatTensor)
            target_var = Variable(density.cuda()).type(torch.cuda.FloatTensor)
            self.optimizer.zero_grad()
            pred_var = self.model(x = input_var, first = self.first)
            if self.first:
                self.first = False
            pred_var = pred_var.clamp(min=-10, max=15)

            loss = self.density_criterion(pred_var, target_var)


            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()


            self.update_recorder(recorder, density_loss=loss, pred_density=pred_var, target_density=target_var)
        self.update_loss(recorder, 'train')
        utility.print_info(recorder, epoch=self.epoch, preffix='Train')

    def validate_st(self):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)
        self.first = True

        result_density = []

        for i, (idx, image, density) in enumerate(self.test_loader):
            print(f'Validating.... {i/num_iter*100:.1f}% \r', end='')

            input_var = torch.autograd.Variable(image.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            density_var = torch.autograd.Variable(density.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            if i == 600:
                self.first = True

            pred_density = self.model(x = input_var, first = self.first)
            if self.first:
                self.first = False

            loss = self.density_criterion(pred_density, density_var)

            for i in range(idx.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[i])

            self.update_recorder(recorder, density_loss=loss, pred_density = pred_density, target_density=density_var)
        self.update_loss(recorder, 'test')
        print('[*** Best MAE: {best_record:.2f} ***]'.format(best_record = self.best_record))
        utility.print_info(recorder, preffix='*** Validation*** ')
        self.save_checkpoint(result_dict={'density':result_density}, recorder=recorder)


    def train_recon(self):
        self.model.train()
        self.epoch += 1
        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(self.train_loader)
        self.current_time = time.time()



        for i, (idx, image, density) in enumerate(self.train_loader):
            print(f'Training...{i/num_iter*100:0.1f}% \r', end='')



            input_var = Variable(image.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)

            self.optimizer.zero_grad()
            pred_var = self.model(input_var)
            pred_var = pred_var.clamp(min=-10, max=15)

            loss = self.recon(pred_var, input_var)


            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()


            self.update_recorder(recorder, recon_loss=loss)
        self.update_loss(recorder, 'train')
        utility.print_info(recorder, epoch=self.epoch, preffix='Train')

    def validate_recon(self):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)

        result_density = []

        for i, (idx, image, density) in enumerate(self.test_loader):
            print(f'Validating.... {i/num_iter*100:.1f}% \r', end='')

            input_var = torch.autograd.Variable(image.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)


            pred_density  = self.model(input_var)

            loss = self.recon(pred_density, input_var)

            for i in range(idx.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[i])

            self.update_recorder(recorder, recon_loss=loss)
        self.update_loss(recorder, 'test')
        utility.print_info(recorder, preffix='*** Validation*** ')
        self.save_checkpoint(result_dict={'density':result_density}, recorder=recorder)

    def train_dfe(self):
        self.model.train()
        self.epoch += 1 
        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(self.train_loader)
        self.current_time = time.time()

        for i, (idx, image, density) in enumerate(self.train_loader):
            print(f'Training...{i/num_iter*100:0.1f}% \r', end='')


            input_var = Variable(image.cuda()).type(torch.cuda.FloatTensor)
            target_var = Variable(density.cuda()).type(torch.cuda.FloatTensor)
            self.optimizer.zero_grad()

            pred_var, recon, crowd_fc, non_fc = self.model(x=input_var, train=True)
            pred_var = pred_var.clamp(min=-10, max=15)

            density_loss = self.density_criterion(pred_var, target_var)
            recon_loss = self.recon(recon, input_var)
    #        diff_loss = self.diff(crowd_fc, non_fc)
            loss = density_loss + 0.1*recon_loss #+ diff_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
            self.optimizer.step()

            self.update_recorder(recorder, density_loss=density_loss, recon_loss=recon_loss, 
                           pred_density=pred_var, target_density = target_var )
        self.update_loss(recorder, 'train')
        utility.print_info(recorder, epoch=self.epoch, preffix='Train')
    def validate_dfe(self):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)

        result_density = []

        for i, (idx, image, density) in enumerate(self.test_loader):
            print(f'Validating....{i/num_iter*100:.1f}% \r', end='')

            image_var = Variable(image.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            density_var = Variable(density.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)

            image_list = []
            result_list = []
            image_list.append(image_var[:,:,0:360,0:360])
            image_list.append(image_var[:,:,0:360,360:720])
            image_list.append(image_var[:,:,360:720,0:360])
            image_list.append(image_var[:,:,360:720,360:720])
            for j in range(4):
                pred_density_, recon, _,_ = self.model(x=image_list[j], train=False)
                result_list.append(pred_density_)
            pred_density = Variable(torch.zeros(1,1,90,90)).cuda()
            pred_density[:,:,0:45,0:45] = result_list[0]
            pred_density[:,:,0:45,45:90] = result_list[1]
            pred_density[:,:,45:90,0:45] = result_list[2]
            pred_density[:,:,45:90,45:90] = result_list[3]
            loss = self.density_criterion(pred_density, density_var)
            
            for j in range(idx.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[j])

            self.update_recorder(recorder, density_loss=loss, pred_density = pred_density, target_density=density_var)

        self.update_loss(recorder, 'test')
        print('[*** Best MAE: {best_record:.2f} ***]'.format(best_record = self.best_record))
        utility.print_info(recorder, preffix='*** Validation *** ')
        self.save_checkpoint(result_dict={'density' : result_density}, recorder=recorder)

        

    def validate_crowd(self):
        self.model.eval()
        recorder = self.init_recorder(self.test_recorder_list)
        self.current_time = time.time()
        num_iter = len(self.test_loader)

        result_density = []

        for i, (idx, image, density) in enumerate(self.test_loader):
            print(f'Validating.... {i/num_iter*100:.1f}% \r', end='')

            input_var = torch.autograd.Variable(image.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)
            density_var = torch.autograd.Variable(density.cuda(), requires_grad=False, volatile=True).type(torch.cuda.FloatTensor)


            pred_density  = self.model(input_var)

            loss = self.density_criterion(pred_density, density_var)

            for i in range(idx.size(0)):
                result_density.append(pred_density.data.cpu().numpy()[i])

            self.update_recorder(recorder, density_loss=loss, pred_density = pred_density, target_density=density_var)
        self.update_loss(recorder, 'test')
        print('[*** Best MAE: {best_record:.2f} ***]'.format(best_record = self.best_record))
        utility.print_info(recorder, preffix='*** Validation*** ')
        self.save_checkpoint(result_dict={'density':result_density}, recorder=recorder)
        
           
      
    def update_recorder(self, recorder, pred_density=None, target_density=None, **kwargs):
        if pred_density is not None and target_density is not None:
            batch_size = pred_density.size(0)
            pred = np.sum(pred_density.data.cpu().numpy(), axis=(1,2,3))
            truth = np.sum(target_density.data.cpu().numpy(), axis=(1,2,3))
            mae = 'error_mae'
            mse = 'error_mse'
            recorder[mae].update(np.mean(np.abs(pred - truth)), batch_size)
            recorder[mse].update(np.mean((pred-truth)**2), batch_size)


        for name, value in kwargs.items():
            batch_size = value.size(0)
            recorder[name].update(value.data[0], batch_size)

        recorder['time'].update(time.time() - self.current_time)
        self.current_time = time.time()

    def update_loss(self, recorder, mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            df_loss = self.train_loss
        else:
            df_loss = self.test_loss

        n = df_loss.shape[0]
        df_loss.loc[n] =[recorder[x].avg for x in df_loss.columns.values]


        with open(self.chk_dir + '/' + mode + '_loss.csv', 'w') as f:
            df_loss.to_csv(f, header=True)

    def train(self):
        self.model.train()
        self.epoch += 1
        self.current_time = time.time()

        recorder = self.init_recorder(self.train_recorder_list)
        num_iter = len(self.train_loader)

        for i, (idx, image, label) in enumerate(self.train_loader):
            print('Training ' + self.args['model']['arch'] + '.....' + f'{i/num_iter*100:0.1f}% \r', end='')

            image_var = Variable(image.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)

            label = label.squeeze()
            label_var = Variable(label.cuda(), requires_grad=False).type(torch.cuda.LongTensor)

            pred_label = self.model(image_var)
            cls_loss  = self.criterion(pred_label, label_var)


            self.optimizer.zero_grad()
            cls_loss.backward()
            self.optimizer.step()

            self.update_recorder(recorder, cls_loss=cls_loss, pred_label=pred_label, target_label=label_var)
        self.update_loss(recorder, mode='train')
        utility.print_info(recorder, epoch=self.epoch, preffix='Train ' + self.args['model']['arch']+' ')

    def validate(self):
        self.model.eval()   
        recorder = self.init_recorder(self.test_recorder_lsit)
        self.current_time = time.time()
        num_iter = len(self.test_loader)


        for i, (idx, image, label) in enumerate(self.test_loader):
            print('Validate ' + self.args['model']['arch'] + '....' + f'{i/num_iter*100:.1f}% \r', end='')

            image_var = Variable(image.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)

            label = label.squeeze()
            label_var = Variable(label.cuda(), requires_grad=False).type(torch.cuda.LongTensor)

            pred_label = self.model(image_var)
            cls_loss  = self.criterion(pred_label, label_var)

            self.update_recorder(recorder, cls_loss=cls_loss, pred_label = pred_label, target_label=label_var)
        self.update_loss(recorder, mode='test')
        utility.print_info(recorder, preffix='*** Validation *** ' + self.args['model']['arch']+' ')
        self.save_checkpoint(recorder)

    def save_checkpoint(self, result_dict, recorder):
        status = {'epoch' : self.epoch,
                    'optimizer' : self.optimizer.state_dict(),
                    'state_dict' : self.model.state_dict()}
        utility.save_checkpoint(self.chk_dir, status, mode='newest')
        utility.save_result(self.chk_dir, result_dict=result_dict, mode='newest', num=10)


        

        current_record = recorder['error_mae'].avg
        best_record = self.best_record

        if current_record < best_record :
            self.best_record = current_record
            string = '------------------------------------------------------[ best record! ]------------------------------------------'
            print(string)
            utility.save_checkpoint(self.chk_dir, status, mode='best')
            utility.save_result(self.chk_dir, result_dict=result_dict, mode='best')
