import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

import models
import transforms as T
import utils
import distutils.util
from functools import partial
from models.segmentation.deconv import FastDeconv

from datasets.cityscapes import *
from datasets.voc import *


import numpy as np
#comment this line out to avoid extra dependence of pycocotools
#from coco_utils import get_coco 

n_iter=0

def get_dataset(name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": ('~/efs/chengxi/VOC/2012/', VOCSegmentation, 21),#torchvision.datasets.VOCSegmentation
        "voc_aug": ('/datasets01/SBDD/072318/', sbd, 21),
        #"coco": ('/datasets01/COCO/022719/', get_coco, 21),
        "cityscapes":('~/efs/chalupkk/datasets/cityscapes/', Cityscapes, 21),
    }
    p, ds_fn, num_classes = paths[name]
    if name=='voc':
        ds = ds_fn(p, image_set=image_set, transforms=transform,year='2012',download=False)

    elif name=='cityscapes':
        ds=ds_fn(p, split=image_set, mode='fine',target_type='semantic',transforms=transform)
    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes


def get_transform(mode,base_size):
    #base_size = 520
    #crop_size = 480
    crop_size=int(480*base_size/520)

    min_size = int((0.5 if mode=='train' else 1.0) * base_size)
    max_size = int((2.0 if mode=='train' else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if mode=='train':
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            target = target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data, target in metric_logger.log_every(data_loader, print_freq, header):
        
        global n_iter
        n_iter = n_iter + 1

        optimizer.zero_grad()

        target = target.to(device)

        #supervised learning
        output = model(data)
        loss = criterion(output, target)
        loss=loss.mean()
    
        #visualization
        segmap = torch.argmax(output['out'], dim=1)


        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if n_iter % args.print_freq == 0:
            if args.tensorboard:                        
                args.writer.add_scalar('SupLoss', loss.item(), n_iter)
                if n_iter % (args.print_freq*100) == 0:
                    grid = torchvision.utils.make_grid(data[:1])
                    grid=(grid-grid.min())/(grid.max()-grid.min()+1e-5)
                    args.writer.add_image('sup images', grid, n_iter)  

                    segmap=args.colormap[segmap[0].detach().cpu().numpy()]
                    segmap=segmap/255.                
                    args.writer.add_image('sup segmaps', segmap.transpose((2,0,1)), n_iter)

                

from collections import OrderedDict
import datetime
def save_path_formatter(args):

    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = [data_folder_name]

    key_map = OrderedDict()
    key_map['model'] = ''
    key_map['aux_loss'] = 'aux'
    key_map['epochs'] = 'ep'
    key_map['lr']=''
    key_map['batch_size']='bs'
    key_map['deconv']='deconv'
    key_map['block'] = 'blk'
    key_map['deconv_iter'] = 'it'
    key_map['eps'] = 'eps'
    key_map['bias'] = 'bias'
    key_map['base_size'] = 'size'
    key_map['pretrained'] = 'pt'
    key_map['pretrained_backbone'] = 'pt_bb'

    for key, key2 in key_map.items():
        value = args_dict[key]
        if key2 is not '':
            folder_string.append('{}:{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return os.path.join('checkpoints',save_path,timestamp)


def main(args):

    args.log_dir = save_path_formatter(args)
    if args.deconv:
        args.deconv = partial(FastDeconv, bias=args.bias, eps=args.eps, n_iter=args.deconv_iter,block=args.block,sampling_stride=args.stride)

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        args.writer = SummaryWriter(args.log_dir,flush_secs=30)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    transform=get_transform(mode='train',base_size=args.base_size)
    dataset, num_classes = get_dataset(args.dataset, "train", transform=transform)

    transform=get_transform(mode='test',base_size=args.base_size)
    dataset_test, _ = get_dataset(args.dataset, "val", transform=transform)
    



    if args.dataset=='cityscapes':
        args.colormap=np.asarray([dataset.classes[i].color for i in range(max(dataset.new_classes)+1)])
    else:
        args.colormap=create_mapillary_vistas_label_colormap()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    #model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,aux_loss=args.aux_loss,pretrained=args.pretrained)
    model = models.segmentation.__dict__[args.model](num_classes=num_classes, aux_loss=args.aux_loss, pretrained=args.pretrained,deconv=args.deconv,pretrained_backbone=args.pretrained_backbone)

    model.to(device)


    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        del checkpoint

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    elif args.device=='cuda':
        model = torch.nn.DataParallel(model).cuda()

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)


    if args.resume:
        total_steps = len(data_loader)* args.start_epoch
        global n_iter
        for i in range(total_steps):
            n_iter = n_iter + 1            
            lr_scheduler.step()

    start_time = time.time()
    for epoch in range(args.start_epoch,args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch,args.print_freq)
        
        if epoch==0 or (epoch+1)%args.eval_freq==0:
            confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)

            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    #'args': args
                },
                #os.path.join(args.log_dir, 'model_{}.pth'.format(epoch)))
                os.path.join(args.log_dir, 'model.pth'))
            
            print(confmat)

            acc_global, acc, iu = confmat.compute()
            acc_global=acc_global.item() * 100
            iu=iu.mean().item() * 100

            if args.tensorboard:
                args.writer.add_scalar('Acc/Test',acc_global,epoch+1)
                args.writer.add_scalar('IOU/Test',iu,epoch+1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    args.writer.close()



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--dataset', default='cityscapes', help='dataset')
    parser.add_argument('--model', default='deeplabv3_resnet50', help='model:deeplabv3_resnet50/fcn_resnet50')
    parser.add_argument('--base-size', default=520, type=int, help='image base size')
    parser.add_argument('--eval-freq', default=1, help='evaluation frequency')

    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--pretrained-backbone', default=False, type=distutils.util.strtobool, help='use pretrained backbone')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tensorboard', default=True, type=distutils.util.strtobool, help='use tensorboard')
    # for deconv
    parser.add_argument('--deconv', default=False, type=distutils.util.strtobool, help='use deconv')
    parser.add_argument('--block', '--num-groups', default=64, type=int, help='block size in deconv')
    parser.add_argument('--deconv-iter', default=5, type=int, help='number of iters in deconv')
    parser.add_argument('--eps', default=1e-5, type=float, help='for regularization')
    parser.add_argument('--bias', default=True, type=distutils.util.strtobool, help='use bias term in deconv')
    parser.add_argument('--stride', default=3, type=int, help='sampling stride in deconv')

    args = parser.parse_args()
    return args



def create_mapillary_vistas_label_colormap():
  """Creates a label colormap used in Mapillary Vistas segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [165, 42, 42],
      [0, 192, 0],
      [196, 196, 196],
      [190, 153, 153],
      [180, 165, 180],
      [102, 102, 156],
      [102, 102, 156],
      [128, 64, 255],
      [140, 140, 200],
      [170, 170, 170],
      [250, 170, 160],
      [96, 96, 96],
      [230, 150, 140],
      [128, 64, 128],
      [110, 110, 110],
      [244, 35, 232],
      [150, 100, 100],
      [70, 70, 70],
      [150, 120, 90],
      [220, 20, 60],
      [255, 0, 0],
      [255, 0, 0],
      [255, 0, 0],
      [200, 128, 128],
      [255, 255, 255],
      [64, 170, 64],
      [128, 64, 64],
      [70, 130, 180],
      [255, 255, 255],
      [152, 251, 152],
      [107, 142, 35],
      [0, 170, 30],
      [255, 255, 128],
      [250, 0, 30],
      [0, 0, 0],
      [220, 220, 220],
      [170, 170, 170],
      [222, 40, 40],
      [100, 170, 30],
      [40, 40, 40],
      [33, 33, 33],
      [170, 170, 170],
      [0, 0, 142],
      [170, 170, 170],
      [210, 170, 100],
      [153, 153, 153],
      [128, 128, 128],
      [0, 0, 142],
      [250, 170, 30],
      [192, 192, 192],
      [220, 220, 0],
      [180, 165, 180],
      [119, 11, 32],
      [0, 0, 142],
      [0, 60, 100],
      [0, 0, 142],
      [0, 0, 90],
      [0, 0, 230],
      [0, 80, 100],
      [128, 64, 64],
      [0, 0, 110],
      [0, 0, 70],
      [0, 0, 192],
      [32, 32, 32],
      [0, 0, 0],
      [0, 0, 0],
      ])

      
def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])



if __name__ == "__main__":
    args = parse_args()
    main(args)




