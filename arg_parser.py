'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from copy import deepcopy
import numpy as np

import distutils.util

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--msg', default=False, type=distutils.util.strtobool, help='display message')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use-gpu', default=torch.cuda.is_available(), type=distutils.util.strtobool, help='Use GPU or not')

    parser.add_argument('-j','--num-workers', default=16, type=int, help='num of fetching threads')
    parser.add_argument('--result-path', default='./results', help='result path')
    parser.add_argument('--checkpoint-path', default='./checkpoints', help='checkpoint path')
    parser.add_argument('--checkpoint-epoch', default=-1, type=int, help='epochs to save checkpoint ')
    parser.add_argument('--print-freq', default=20, type=int,  help='print freq')


    #important settings:
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--optimizer', default='SGD', help='optimizer(SGD|Adam|AMSGrad)')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', help='learning rate scheduler(multistep|cosine)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('-b','--batch-size', default=128, type=int, help='batch size')

    parser.add_argument('--epochs', default=20, type=int, help='training epochs')
    parser.add_argument('--milestone', default=0.4, type=float, help='milestone in multistep scheduler')
    parser.add_argument('--multistep-gamma', default=0.1, type=float, help='the gamma parameter in multistep|plateau scheduler')

    parser.add_argument('-a','--arch', default='vgg11', help='architecture')
    parser.add_argument('--dataset', default='cifar10', help='dataset(cifar10|cifar100|svhn|stl10|mnist)')

    parser.add_argument('--init', default='kaiming_1', help='initialization method (casnet|xavier|kaiming_1||kaiming_2)')

    parser.add_argument('--save-plot', default=True, type=distutils.util.strtobool,  help='save plots with matplotlib')

    parser.add_argument('--tensorboard', default=True, type=distutils.util.strtobool, help='use tensorboard')
    parser.add_argument('--loss', default='CE', type=str, help='loss: CE/L2')
    parser.add_argument('--method', default=3, type=int, help='method/model type')
    parser.add_argument('--batchnorm', default=True, type=distutils.util.strtobool, help='turns on or off batch normalization')

    # for deconv
    parser.add_argument('--deconv', default=False, type=distutils.util.strtobool, help='use deconv')
    parser.add_argument('--delinear', default=True, type=distutils.util.strtobool, help='use decorrelated linear')

    parser.add_argument('--block-fc','--num-groups-final', default=0, type=int, help='number of groups in the fully connected layers')
    parser.add_argument('--block', '--num-groups', default=64,type=int, help='block size in deconv')
    parser.add_argument('--deconv-iter', default=5,type=int, help='number of iters in deconv')
    parser.add_argument('--eps', default=1e-5,type=float, help='for regularization')
    parser.add_argument('--bias', default=True,type=distutils.util.strtobool, help='use bias term in deconv')
    parser.add_argument('--stride', default=3, type=int, help='sampling stride in deconv')
    parser.add_argument('--freeze', default=False, type=distutils.util.strtobool, help='freeze the deconv updates')

    args = parser.parse_args()

    return args
