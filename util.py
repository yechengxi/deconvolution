import os
import sys
import time
import math
import torch

import torch.nn as nn
import torch.nn.init as init
import numpy as np


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


from collections import OrderedDict
import datetime


def save_path_formatter(args):

    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = [data_folder_name]

    key_map = OrderedDict()
    key_map['arch'] =''

    key_map['epochs'] = 'ep'
    key_map['optimizer']=''
    key_map['lr']=''
    key_map['lr_scheduler']=''
    key_map['batch_size']='bs'
    #key_map['n_iter']='it'
    #key_map['seed']='seed'
    key_map['weight_decay']='wd'
    key_map['batchnorm'] = 'bn'

    key_map['deconv']='deconv'
    key_map['delinear']='delinear'
    key_map['block']='b'
    key_map['stride']='stride'
    key_map['deconv_iter'] = 'it'
    key_map['eps'] = 'eps'
    key_map['bias'] = 'bias'
    key_map['block_fc']='bfc'
    #key_map['freeze']='freeze'


    for key, key2 in key_map.items():
        value = args_dict[key]
        if key2 is not '':
            folder_string.append('{}.{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M")
    return os.path.join('checkpoints',save_path,timestamp).replace("\\","/")



def tensor2array(tensor, max_value=None, colormap='rainbow'):
    if max_value is None:
        tensor=(tensor-tensor.min())/(tensor.max()-tensor.min()+1e-6)
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (tensor.squeeze().numpy()*255./max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5

    #for tensorboardx 1.4
    #array=array.transpose(2,0,1)

    return array

def targets_to_one_hot(targets,num_classes):

    return  torch.zeros(targets.shape[0], num_classes,device=targets.device).scatter_(1, targets.view(-1, 1), 1)

