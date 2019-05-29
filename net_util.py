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



def set_parameters(opts):
    '''
    This function is called before training/testing to set parameters
    :param opts:
    :return opts:
    '''

    if not opts.__contains__('train_losses'):
        opts.train_losses=[]

    if not opts.__contains__('train_accuracies'):
        opts.train_accuracies = []

    if not opts.__contains__('valid_losses'):
        opts.valid_losses = []
    if not opts.__contains__('valid_accuracies'):
        opts.valid_accuracies = []

    if not opts.__contains__('test_losses'):
        opts.test_losses = []
    if not opts.__contains__('test_accuracies'):
        opts.test_accuracies = []

    if not opts.__contains__('best_acc'):
        opts.best_acc = 0.0

    if not opts.__contains__('lowest_loss'):
        opts.lowest_loss = 1e4

    if not opts.__contains__('checkpoint_path'):
        opts.checkpoint_path = 'checkpoints'

    if not os.path.exists(opts.checkpoint_path):
        os.mkdir(opts.checkpoint_path)

    if not opts.__contains__('checkpoint_epoch'):
        opts.checkpoint_epoch = 5








def train_net(net,opts):

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    print('training at epoch {}'.format(opts.epoch+1))

    net.train()

    total_time=0
    data_time=0
    total=1e-3
    correct=0
    extra=0.

    optimizer=opts.current_optimizer

    end_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(opts.data_loader):
        #ff

        if opts.use_gpu:
            targets = targets.cuda()

        data_time += (time.time() - end_time)#loading time

        outputs = net(inputs)

        maps=None
        if type(outputs) is list:
            maps=outputs
            outputs=outputs[-1]

        #loss = opts.criterion(outputs, targets).mean()

        if opts.loss=='CE':
            loss = opts.criterion(outputs, targets)#.mean()
        elif opts.loss=='L2':
            from util import targets_to_one_hot
            targets_one_hot=targets_to_one_hot(targets,opts.num_outputs)
            loss = opts.criterion(outputs, targets_one_hot)#.mean()
        losses.update(loss.item(), inputs.size(0))

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1[0].item(), inputs.size(0))
        top5.update(prec5[0].item(), inputs.size(0))



        #bp
        loss.backward()


        if opts.lr_scheduler == 'cosine':
            opts.current_scheduler.step()
        optimizer.step()


        if opts.tensorboard and (maps is not None):

            if batch_idx % opts.viz_T == 0:
                # Log images (image summary)

                for l in range(len(maps) - 1):
                    c, h, w = maps[l][0].shape
                    tmp = maps[l][0].detach().permute([1, 0, 2]).contiguous().view(h, w * c).cpu()
                    opts.writer.image_summary('train maps {}'.format(l),
                                              [tensor2array(tmp, max_value=None, colormap='bone')],
                                              opts.logger_n_iter)

                # Log values and gradients of the parameters (histogram summary)

                for tag, value in net.named_parameters():
                    #print(tag)
                    tag = tag.replace('.', '/')
                    opts.writer.histo_summary(tag, value.data.cpu().numpy(), opts.logger_n_iter)
                    if hasattr(value.grad, 'data'):
                        opts.writer.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), opts.logger_n_iter)

        if opts.tensorboardX:
            if opts.logger_n_iter%opts.viz_T==0:
                opts.writer.add_scalar('loss', loss.item(), opts.logger_n_iter )

        opts.logger_n_iter += 1

        optimizer.zero_grad()  # flush

        total_time += (time.time() - end_time)
        end_time = time.time()


        if opts.msg:
            print('Loss: %.3f | top1: %.3f%% ,top5: %.3f%%'
                  % (losses.avg, top1.avg, top5.avg))




        opts.train_batch_logger.log({
            'epoch': (opts.epoch+1),
            'batch': batch_idx + 1,
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg,
            'time': total_time,

        })


    opts.train_epoch_logger.log({
        'epoch': (opts.epoch+1),
        'loss': losses.avg,
        'top1': top1.avg,
        'top5':top5.avg,
        'time': total_time,
    })

    print('Loss: %.3f | top1: %.3f%%, top5: %.3f%% elasped time: %3.f seconds.'
          % (losses.avg,  top1.avg, top5.avg, total_time))
    opts.train_accuracies.append(top1.avg)

    opts.train_losses.append(losses.avg)




def eval_net(net,opts):

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    if opts.validating:
        print('Validating at epoch {}'.format(opts.epoch + 1))

    if opts.testing:
        print('Testing at epoch {}'.format(opts.epoch + 1))


    if not opts.__contains__('validating'):
        opts.validating = False
    if not opts.__contains__('testing'):
        opts.testing = False


    net.eval()
    total = 1e-3
    total_time = 0



    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(opts.data_loader):

        with torch.no_grad():

            if opts.use_gpu:
                targets = targets.cuda()

            outputs = net(inputs)
            if type(outputs) is list:
                outputs = outputs[-1]

            if opts.loss == 'CE':
                loss = opts.criterion(outputs, targets)  # .mean()
            elif opts.loss == 'L2':
                from util import targets_to_one_hot
                targets_one_hot = targets_to_one_hot(targets, opts.num_outputs)
                loss = opts.criterion(outputs, targets_one_hot)  # .mean()

            losses.update(loss.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0].item(), inputs.size(0))
            top5.update(prec5[0].item(), inputs.size(0))

            total_time += (time.time() - end_time)
            end_time = time.time()

            if opts.msg:
                print('Loss: %.3f | top1: %.3f%% ,top5: %.3f%%'
                      % (losses.avg, top1.avg, top5.avg))



    if  opts.testing:
        opts.test_losses.append(losses.avg)
        opts.test_accuracies.append(top1.avg)

        opts.test_epoch_logger.log({
            'epoch': (opts.epoch + 1),
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg,
            'time': total_time,
        })

    if  opts.validating:
        opts.valid_losses.append(losses.avg)
        opts.valid_accuracies.append(top1.avg)

        opts.valid_epoch_logger.log({
            'epoch': (opts.epoch + 1),
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg,
            'time': total_time,
        })
    # Save checkpoint.

    is_best=(top1.avg > opts.best_acc)
    if is_best:
        opts.best_acc = top1.avg

    states = {
        'state_dict': net.module.state_dict() if hasattr(net,'module') else net.state_dict(),
        'epoch': opts.epoch+1,
        'arch': opts.arch,
        'best_acc': opts.best_acc,
        'train_losses': opts.train_losses,
        'optimizer': opts.current_optimizer.state_dict()
    }


    if opts.__contains__('acc'):
        states['acc']=top1.avg,

    if opts.__contains__('valid_losses'):
        states['valid_losses']=opts.valid_losses
    if opts.__contains__('test_losses'):
        states['test_losses'] = opts.test_losses


    if (opts.checkpoint_epoch > 0):
        if not os.path.isdir(opts.checkpoint_path):
            os.mkdir(opts.checkpoint_path)


    save_file_path = os.path.join(opts.checkpoint_path, 'checkpoint.pth.tar')
    torch.save(states, save_file_path)

    if is_best:
        shutil.copyfile(save_file_path, os.path.join(opts.checkpoint_path,'model_best.pth.tar'))


    print('Loss: %.3f | top1: %.3f%%, top5: %.3f%%, elasped time: %3.f seconds. Best Acc: %.3f%%'
          % (losses.avg , top1.avg, top5.avg, total_time, opts.best_acc))




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