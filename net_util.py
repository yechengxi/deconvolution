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
from torch.autograd import Variable

from util import tensor2array
import shutil



def set_parameters(args):
    '''
    This function is called before training/testing to set parameters
    :param args:
    :return args:
    '''

    if not args.__contains__('train_losses'):
        args.train_losses=[]

    if not args.__contains__('train_accuracies'):
        args.train_accuracies = []

    if not args.__contains__('valid_losses'):
        args.valid_losses = []
    if not args.__contains__('valid_accuracies'):
        args.valid_accuracies = []

    if not args.__contains__('test_losses'):
        args.test_losses = []
    if not args.__contains__('test_accuracies'):
        args.test_accuracies = []

    if not args.__contains__('best_acc'):
        args.best_acc = 0.0

    if not args.__contains__('lowest_loss'):
        args.lowest_loss = 1e4

    if not args.__contains__('checkpoint_path'):
        args.checkpoint_path = 'checkpoints'

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not args.__contains__('checkpoint_epoch'):
        args.checkpoint_epoch = 5




def train_net(net,args):

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    print('training at epoch {}'.format(args.epoch+1))

    net.train()

    total_time=0
    data_time=0
    total=1e-3
    correct=0
    extra=0.

    optimizer=args.current_optimizer

    end_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(args.data_loader):
        #ff
        if args.use_gpu:
            targets = targets.cuda()

        data_time += (time.time() - end_time)#loading time

        outputs = net(inputs)

        maps=None
        if type(outputs) is list:
            maps=outputs
            outputs=outputs[-1]

        #loss = args.criterion(outputs, targets).mean()

        if args.loss=='CE':
            loss = args.criterion(outputs, targets)#.mean()
        elif args.loss=='L2':
            from util import targets_to_one_hot
            targets_one_hot=targets_to_one_hot(targets,args.num_outputs)
            loss = args.criterion(outputs, targets_one_hot)*args.num_outputs*0.5
        losses.update(loss.item(), inputs.size(0))

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1[0].item(), inputs.size(0))
        top5.update(prec5[0].item(), inputs.size(0))



        #bp
        loss.backward()

        optimizer.step()
        
        if args.lr_scheduler == 'cosine':
            args.current_scheduler.step()
            
        if args.tensorboard:
            if args.logger_n_iter%args.print_freq==0:
                args.writer.add_scalar('loss', loss.item(), args.logger_n_iter )

        args.logger_n_iter += 1
        optimizer.zero_grad()  # flush
        total_time += (time.time() - end_time)
        end_time = time.time()

        if args.msg:
            print('Loss: %.3f | top1: %.3f%% ,top5: %.3f%%'
                  % (losses.avg, top1.avg, top5.avg))


        args.train_batch_logger.log({
            'epoch': (args.epoch+1),
            'batch': batch_idx + 1,
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg,
            'time': total_time,

        })


    args.train_epoch_logger.log({
        'epoch': (args.epoch+1),
        'loss': losses.avg,
        'top1': top1.avg,
        'top5':top5.avg,
        'time': total_time,
    })

    print('Loss: %.3f | top1: %.3f%%, top5: %.3f%% elasped time: %3.f seconds.'
          % (losses.avg,  top1.avg, top5.avg, total_time))
    args.train_accuracies.append(top1.avg)

    args.train_losses.append(losses.avg)




def eval_net(net,args):

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    if args.validating:
        print('Validating at epoch {}'.format(args.epoch + 1))

    if args.testing:
        print('Testing at epoch {}'.format(args.epoch + 1))


    if not args.__contains__('validating'):
        args.validating = False
    if not args.__contains__('testing'):
        args.testing = False


    net.eval()
    total = 1e-3
    total_time = 0



    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(args.data_loader):

        with torch.no_grad():

            if args.use_gpu:
                targets = targets.cuda()

            outputs = net(inputs)
            if type(outputs) is list:
                outputs = outputs[-1]

            if args.loss == 'CE':
                loss = args.criterion(outputs, targets)  # .mean()
            elif args.loss == 'L2':
                from util import targets_to_one_hot
                targets_one_hot = targets_to_one_hot(targets, args.num_outputs)
                loss = args.criterion(outputs, targets_one_hot)*args.num_outputs*0.5

            losses.update(loss.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0].item(), inputs.size(0))
            top5.update(prec5[0].item(), inputs.size(0))

            total_time += (time.time() - end_time)
            end_time = time.time()

            if args.msg:
                print('Loss: %.3f | top1: %.3f%% ,top5: %.3f%%'
                      % (losses.avg, top1.avg, top5.avg))



    if  args.testing:
        args.test_losses.append(losses.avg)
        args.test_accuracies.append(top1.avg)

        args.test_epoch_logger.log({
            'epoch': (args.epoch + 1),
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg,
            'time': total_time,
        })

    if  args.validating:
        args.valid_losses.append(losses.avg)
        args.valid_accuracies.append(top1.avg)

        args.valid_epoch_logger.log({
            'epoch': (args.epoch + 1),
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg,
            'time': total_time,
        })
    # Save checkpoint.

    is_best=(top1.avg > args.best_acc)
    if is_best:
        args.best_acc = top1.avg

    states = {
        'state_dict': net.module.state_dict() if hasattr(net,'module') else net.state_dict(),
        'epoch': args.epoch+1,
        'arch': args.arch,
        'best_acc': args.best_acc,
        'train_losses': args.train_losses,
        'optimizer': args.current_optimizer.state_dict()
    }


    if args.__contains__('acc'):
        states['acc']=top1.avg,

    if args.__contains__('valid_losses'):
        states['valid_losses']=args.valid_losses
    if args.__contains__('test_losses'):
        states['test_losses'] = args.test_losses


    if (args.checkpoint_epoch > 0):
        if not os.path.isdir(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)


    save_file_path = os.path.join(args.checkpoint_path, 'checkpoint.pth.tar')
    torch.save(states, save_file_path)

    if is_best:
        shutil.copyfile(save_file_path, os.path.join(args.checkpoint_path,'model_best.pth.tar'))


    print('Loss: %.3f | top1: %.3f%%, top5: %.3f%%, elasped time: %3.f seconds. Best Acc: %.3f%%'
          % (losses.avg , top1.avg, top5.avg, total_time, args.best_acc))




import csv

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res