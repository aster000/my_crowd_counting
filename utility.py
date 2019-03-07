import json
import numpy as np
import h5py
import time
import os, sys
from shutil import copyfile
import torch





class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def print_info(recorder, epoch=None,  mode = None, preffix='', suffix=''):
    str = preffix
    if epoch is not None:
        str = str + 'Epoch: {epoch}    '.format(epoch=epoch)
    if 'time' in recorder:
        str = str + 'Time: {time.sum:.3f}   '.format(time=recorder['time'])
    if 'density_loss' in recorder:
        str = str + 'density_loss: {density_loss.avg:0.5f}  '.format(density_loss=recorder['density_loss'])
        
    if 'error_mae' in recorder and 'error_mse' in recorder:
        rmse = np.sqrt(recorder['error_mse'].avg)
        str = str + ' Error [{error_mae.avg:.3f} {mse:.3f}]'.format(error_mae=recorder['error_mae'], mse=rmse)

    str = str + suffix
    print(str)

def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)


def save_args(chk_dir, args):
    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)

    os.makedirs(chk_dir + '/my_models')

    file_list = args['project_files']
    
    for f in file_list :
        copyfile(f, chk_dir + '/' + f)

class Timer(object):
    def __init__(self, description=""):
        self.description = description

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.begin = time.time()

    def stop(self):
        self.end = time.time()
        self.interval = self.end - self.begin
        print(f'{self.description} cost time: {self.interval:0.2f}s')

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __def__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)

        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()

def save_result(chk_dir, result_dict, mode='newest', num=None):
    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)

    result_file = chk_dir + '/' + mode + '_result.h5'

    with h5py.File(result_file, 'w') as hdf:
        for key, values in result_dict.items():
            n = len(values) if num is None else num
            for i in range(n):
                hdf.create_dataset(key + '/' + str(i), data=values[i])


def save_checkpoint(chk_dir, stats, mode='newest'):

    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)

    chk_file = chk_dir + '/' + mode + '_checkpoint.tar'
    torch.save(stats, chk_file)
