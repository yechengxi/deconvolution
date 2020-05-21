import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models.deconv import *

import distutils.util
from functools import partial

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data',metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='cosine', help='learning rate scheduler(multistep|cosine)')
parser.add_argument('--scheduler-step-size', default=30, type=int, help='step size in StepLR scheduler')

parser.add_argument('--milestone', default=0.3, type=float, help='milestone in multistep scheduler')
parser.add_argument('--multistep-gamma', default=0.1, type=float,
                    help='the gamma parameter in multistep|plateau scheduler')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dataset', default='imagenet', help='dataset')

parser.add_argument('--tensorboard', default=True, type=distutils.util.strtobool, help='use tensorboard')
parser.add_argument('--save-plot', default=True, type=distutils.util.strtobool, help='save plots with matplotlib')

# for deconv
parser.add_argument('--deconv', default=False, type=distutils.util.strtobool, help='use deconv')
parser.add_argument('--delinear', default=True, type=distutils.util.strtobool, help='use decorrelated linear functions')
parser.add_argument('--block', '--num-groups', default=64, type=int, help='block size in deconv')
parser.add_argument('--deconv-iter', default=5, type=int, help='number of iters in deconv')
parser.add_argument('--eps', default=1e-5, type=float, help='for regularization')
parser.add_argument('--bias', default=True, type=distutils.util.strtobool, help='use bias term in deconv')
parser.add_argument('--block-fc','--num-groups-final', default=64, type=int, help='block number in the fully connected layers.')
parser.add_argument('--test-run', default=False, type=distutils.util.strtobool, help='test run only')
parser.add_argument('--test-iter', default=500, type=int, help='test iterations')
parser.add_argument('--stride', default=3, type=int, help='sampling stride in deconv')
#parser.add_argument('--freeze', default=False, type=distutils.util.strtobool, help='freeze the deconv updates')

best_acc1 = 0
n_iter = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.log_dir = save_path_formatter(args)
    if args.deconv:
        args.deconv = partial(FastDeconv, bias=args.bias, eps=args.eps, n_iter=args.deconv_iter,block=args.block,sampling_stride=args.stride)

    if args.delinear:
        args.channel_deconv=None
        if args.block_fc > 0:
            args.delinear = partial(Delinear, block=args.block_fc, eps=args.eps,n_iter=args.deconv_iter)
        else:
            args.delinear = None
    else:
        args.delinear = None
        if args.block_fc > 0:
            args.channel_deconv = partial(ChannelDeconv, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter,sampling_stride=args.stride)
        else:
            args.channel_deconv = None


    args.train_losses=[]
    args.train_top1=[]
    args.train_top5=[]
    args.eval_losses=[]
    args.eval_top1=[]
    args.eval_top5=[]
    args.cur_losses=[]

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        args.writer = SummaryWriter(args.log_dir,flush_secs=30)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch in models.__dict__:
            model = models.__dict__[args.arch]()
        elif args.arch=='resnet18d':
            from models.resnet_imagenet import resnet18d
            model = resnet18d(deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
        elif args.arch == 'resnet34d':
            from models.resnet_imagenet import resnet34d
            model = resnet34d(deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
        elif args.arch == 'resnet50d':
            from models.resnet_imagenet import resnet50d
            model = resnet50d(deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
        elif args.arch == 'resnet101d':
            from models.resnet_imagenet import resnet101d
            model = resnet101d(deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
        elif args.arch=='vgg11d':
            from models.vgg_imagenet import vgg11d
            model = vgg11d('VGG11d', deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
        elif args.arch == 'vgg16d':
            from models.vgg_imagenet import vgg16d
            model = vgg16d('VGG16d', deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
        elif args.arch == 'densenet121d':
            from models.densenet_imagenet import densenet121d
            model = densenet121d(deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print(args)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params,'trainable parameters in the network.')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.evaluate:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        validate(val_loader, model, criterion, 0, args)
        return

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.lr_scheduler=='multistep':

        milestones=[int(args.milestone*args.epochs)]
        while milestones[-1]+milestones[0]<args.epochs:
            milestones.append(milestones[-1]+milestones[0])
        args.current_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.multistep_gamma)

    if args.lr_scheduler=='step':
        args.current_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, gamma=args.multistep_gamma)

    if args.lr_scheduler=='cosine':
        total_steps = math.ceil(len(train_dataset)/args.batch_size)*args.epochs
        args.current_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0, last_epoch=-1)

    if args.resume:
        lr = args.lr
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
        if args.lr_scheduler == 'multistep' or args.lr_scheduler == 'step':
            for i in range(args.start_epoch):
                args.current_scheduler.step()
        if args.lr_scheduler == 'cosine':
            total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.start_epoch
            global n_iter
            for i in range(total_steps):
                n_iter = n_iter + 1
                args.current_scheduler.step()


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)
        if args.lr_scheduler == 'multistep' or args.lr_scheduler =='step':
            args.current_scheduler.step()
        if args.lr_scheduler == 'multistep' or args.lr_scheduler =='step' or args.lr_scheduler == 'cosine':
            print('Current learning rate:', args.current_scheduler.get_lr()[0])

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)


        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)


        if args.save_plot:
            plt.subplot(1, 3, 1)
            plt.title('Loss Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(args.train_losses, 'b')
            plt.plot(args.eval_losses, 'r')


            plt.subplot(1, 3, 2)
            plt.title('Top 1 Accuracy Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Top 1 Acc', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(args.train_top1, 'b')
            plt.plot(args.eval_top1, 'r')

            plt.subplot(1, 3, 3)
            plt.title('Top 5 Accuracy Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Top 5 Acc', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(args.train_top5, 'b')
            plt.plot(args.eval_top5, 'r')

            plt.savefig(os.path.join(args.log_dir, 'TrainingPlots'))
            plt.clf()


        #if args.test_run:
        #    break


        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,path=args.log_dir)

    args.writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if args.test_run and i>args.test_iter:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_scheduler == 'cosine':
            args.current_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        global n_iter
        n_iter = n_iter + 1

        if i % args.print_freq == 0:
            progress.print(i)
            if args.tensorboard:
                args.writer.add_scalar('CurrentLoss', losses.val, n_iter)



    if args.tensorboard:
        args.writer.add_scalar('Loss/Train',losses.avg,epoch+1)
        args.writer.add_scalar('Prec/Train1',top1.avg,epoch+1)
        args.writer.add_scalar('Prec/Train5', top5.avg, epoch + 1)

    args.train_losses.append(losses.avg)
    args.train_top1.append(top1.avg)
    args.train_top5.append(top5.avg)


def validate(val_loader, model, criterion,epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.test_run and i > args.test_iter:
                break
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if args.tensorboard:
            args.writer.add_scalar('Loss/Val',losses.avg,epoch + 1)
            args.writer.add_scalar('Prec/Val1',top1.avg,epoch + 1)
            args.writer.add_scalar('Prec/Val5', top5.avg, epoch + 1)

        args.eval_losses.append(losses.avg)
        args.eval_top1.append(top1.avg)
        args.eval_top5.append(top5.avg)

    return top1.avg


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path,filename))
    if is_best:
        shutil.copyfile(os.path.join(path,filename), os.path.join(path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


from collections import OrderedDict
import datetime

def save_path_formatter(args):

    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = [data_folder_name]

    key_map = OrderedDict()
    key_map['arch'] = ''
    key_map['epochs'] = 'ep'
    #key_map['optimizer']=''
    key_map['lr']=''
    key_map['lr_scheduler']=''
    key_map['weight_decay'] = 'wd'
    key_map['batch_size']='bs'
    key_map['seed']='seed'
    key_map['deconv'] = 'deconv'
    key_map['delinear'] = 'delinear'
    key_map['stride']='stride'
    key_map['block'] = 'b'
    key_map['deconv_iter'] = 'it'
    key_map['eps'] = 'eps'
    key_map['bias'] = 'bias'
    key_map['block_fc']='bfc'


    for key, key2 in key_map.items():
        value = args_dict[key]
        if key2 is not '':
            folder_string.append('{}:{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return os.path.join('checkpoints',save_path,timestamp)

if __name__ == '__main__':
    main()