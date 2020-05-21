'''Train  with PyTorch.'''
from __future__ import print_function

import torch.nn as nn

from models import *

from net_util import *

from arg_parser import *

cudnn.benchmark=True

from functools import partial
import csv


if __name__ == '__main__':

    args=parse_args()

    from util import save_path_formatter
    log_dir=save_path_formatter(args)
    args.checkpoint_path=log_dir
    args.result_path=log_dir
    args.log_path=log_dir

    if args.save_plot:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

    if args.deconv:
        args.deconv = partial(FastDeconv,bias=args.bias, eps=args.eps, n_iter=args.deconv_iter,block=args.block,sampling_stride=args.stride)
    else:
        args.deconv=None

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


    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    args.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    if args.dataset=='cifar10':
        args.in_planes = 3
        args.input_size=32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        args.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        print("| Preparing CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        args.num_outputs = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif (args.dataset == 'cifar100'):
        args.in_planes = 3
        args.input_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])

        print("| Preparing CIFAR-100 dataset...")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        args.num_outputs = 100


    elif args.dataset=='mnist':
        args.in_planes=1
        args.input_size = 28
        trainset= torchvision.datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        testset=torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        args.num_outputs = 10
    elif args.dataset == 'fashion':
        args.in_planes = 1
        args.input_size = 28
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                              transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
        args.num_outputs = 10
    elif args.dataset=='stl10':
        args.in_planes = 3
        args.input_size = 96
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        args.classes=('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

        print("| Preparing STL10 dataset...")

        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=True, transform=transform_test)
        args.num_outputs = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif args.dataset=='svhn':
        args.in_planes = 3
        args.input_size = 32
        trainset = torchvision.datasets.SVHN(root='./data',  split='train', download=True,
                                              transform=transforms.Compose([
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.3782,  0.3839,  0.4100),(0.1873,  0.1905,  0.1880))
                                              ]))

        testset = torchvision.datasets.SVHN(root='./data',  split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3782, 0.3839, 0.4100), (0.1873, 0.1905, 0.1880))
        ]))
        args.num_outputs = 10
    elif (args.dataset == 'imagenet16') or (args.dataset == 'imagenet32'):
        args.in_planes = 3
        if (args.dataset == 'imagenet16'):
            args.input_size = 16
            datapath='./data/Imagenet16'
        if (args.dataset == 'imagenet32'):
            args.input_size = 32
            datapath = './data/Imagenet32'

        transform_train = transforms.Compose([
            transforms.RandomCrop(args.input_size, padding=int(math.log2(args.input_size)-1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        print("| Preparing ImageNet-{} dataset...".format(args.input_size))
        from imagenet_loader import *
        trainset = ImageNetDS(datapath, img_size=args.input_size, train=True, transform=transform_train, target_transform=None)
        testset = ImageNetDS(datapath, img_size=args.input_size, train=False, transform=transform_test, target_transform=None)
        args.num_outputs = 1000


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers)


    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    args.train_epoch_logger = Logger(os.path.join(args.result_path, 'train.log'),
                                  ['epoch','loss', 'top1', 'top5','time'])
    args.train_batch_logger = Logger(os.path.join(args.result_path, 'train_batch.log'),
                                     ['epoch','batch', 'loss', 'top1', 'top5', 'time'])
    args.test_epoch_logger = Logger(os.path.join(args.result_path, 'test.log'),
                                    ['epoch', 'loss', 'top1', 'top5', 'time'])

    # Model

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        args.writer = SummaryWriter(args.log_path,flush_secs=20)

    print('==> Building model..')

    if args.deconv:
        args.batchnorm=False
        print('************ Batch norm disabled when deconv is used. ************')

    if (not args.deconv) and args.channel_deconv:
        print('************ Channel Deconv is used on the original model, this accelrates the training. If you want to turn it off set --num-groups-final 0 ************')


    if args.arch == 'vgg19':
        net = VGG('VGG19',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'vgg16':
        net = VGG('VGG16',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'vgg13':
        net = VGG('VGG13',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'vgg11':
        net = VGG('VGG11',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'vggx':
        from models.simple import VGGX
        net = VGGX('VGGX',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'vgg11d':
        from models.vgg_imagenet import vgg11d
        net = vgg11d('VGG11d',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'vgg16d':
        from models.vgg_imagenet import vgg16d
        net = vgg16d('VGG16d',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch=='resnet':
        net = ResNet18(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch=='resnet18d':
        from models.resnet_imagenet import resnet18d
        net = resnet18d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'resnet34d':
        from models.resnet_imagenet import resnet34d
        model = resnet34d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    if args.arch == 'resnet50d':
        from models.resnet_imagenet import resnet50d
        model = resnet50d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch=='resnet34':
        net = ResNet34(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch=='resnet50':
        net = ResNet50(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch=='preact':
        net = PreActResNet18(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    # net = GoogLeNet()
    if args.arch == 'densenet':
        net = densenet_cifar()

    if args.arch == 'densenet121':
        net = DenseNet121(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'densenet121d':
        from models.densenet_imagenet import densenet121d
        net = densenet121d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'simple_v1':
        from models.simple import *
        net = SimpleCNN_v1(channels_in=args.in_planes,kernel_size=args.input_size,num_outputs=args.num_outputs,method=args.method)

    if args.arch == 'simple_v2':
        from models.simple import *
        net = SimpleCNN_v2(channels_in=args.in_planes,kernel_size=3,hidden_layers=10,hidden_channels=4,num_outputs=args.num_outputs,method=args.method)

    if args.arch == 'mlp':
        from models.simple import *
        net = MLP(input_nodes=784, hidden_nodes=128,hidden_layers=3,method=args.method,num_outputs=args.num_outputs)

    if args.arch == 'efficient':
        from models.efficientnet import *
        net = EfficientNetB0(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'resnext':
        from models.resnext import *
        net = ResNeXt29_32x4d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'mobilev2':
        from models.mobilenetv2 import *
        net = MobileNetV2(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    # net = MobileNet()
    if args.arch == 'dpn':
        net = DPN92(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    # net = ShuffleNetG2()

    if args.arch == 'senet':
        net = SENet18(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch == 'pnasnetA':
        net = PNASNetA(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    if args.arch == 'pnasnetB':
        net = PNASNetB(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.arch=='lenet':
        net = LeNet(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if args.loss=='CE':
        args.criterion = nn.CrossEntropyLoss()
        if args.use_gpu:
            args.criterion = nn.CrossEntropyLoss().cuda()
            #args.criterion = torch.nn.DataParallel(args.criterion)
    elif args.loss=='L2':
        args.criterion = nn.MSELoss()
        if args.use_gpu:
            args.criterion = nn.MSELoss().cuda()
            # args.criterion = torch.nn.DataParallel(args.criterion)


    args.logger_n_iter = 0


    # Training


    print(args)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params,'trainable parameters in the network.')

    set_parameters(args)

    lr = args.lr

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    if args.optimizer == 'SGD':
        args.current_optimizer = optim.SGD(parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        args.current_optimizer = optim.Adam(parameters, lr=lr, weight_decay=args.weight_decay)

    if args.lr_scheduler=='multistep':

        milestones=[int(args.milestone*args.epochs)]
        while milestones[-1]+milestones[0]<args.epochs:
            milestones.append(milestones[-1]+milestones[0])

        args.current_scheduler = optim.lr_scheduler.MultiStepLR(args.current_optimizer, milestones=milestones, gamma=args.multistep_gamma)

    if args.lr_scheduler=='cosine':
        total_steps = math.ceil(len(trainset)/args.batch_size)*args.epochs
        args.current_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(args.current_optimizer, total_steps, eta_min=0, last_epoch=-1)

    args.total_steps = math.ceil(len(trainset)/args.batch_size)*args.epochs
    args.cur_steps=0

    if args.use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    device = torch.device("cuda")

    plotting_accuracies = []

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_acc = checkpoint['best_acc']
            if hasattr(net,'module'):
                net.module.load_state_dict(checkpoint['state_dict'])
            else:
                net.load_state_dict(checkpoint['state_dict'])
            args.current_optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.resume:
        lr = args.lr
        for param_group in args.current_optimizer.param_groups:
            param_group['lr'] = lr
        if args.lr_scheduler == 'multistep':
            for i in range(args.start_epoch):
                args.current_scheduler.step()
        if args.lr_scheduler == 'cosine':
            total_steps = math.ceil(len(trainset) / args.batch_size) * args.start_epoch
            for i in range(total_steps):
                args.current_scheduler.step()



    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        args.epoch = epoch
        if args.lr_scheduler == 'multistep':
            args.current_scheduler.step()
        if args.lr_scheduler == 'multistep' or args.lr_scheduler=='cosine':
            print('Current learning rate:', args.current_scheduler.get_lr()[0])



        args.data_loader = train_loader
        train_net(net,args)
        args.data_loader = test_loader

        args.validating=False
        args.testing=True

        eval_net(net,args)

        if args.tensorboard:
            #Log scalar values (scalar summary)
            losses={'train':args.train_losses[-1]}
            args.writer.add_scalar('Loss/train', args.train_losses[-1], epoch + 1)

            if len(args.valid_losses) > 0:
                losses['valid'] = args.valid_losses[-1]
                args.writer.add_scalar('Loss/valid', args.valid_losses[-1], epoch + 1)

            if len(args.test_losses) > 0:
                losses['test'] = args.test_losses[-1]
                args.writer.add_scalar('Loss/test', args.test_losses[-1], epoch + 1)
                
            accuracies={'train':args.train_accuracies[-1]}

            args.writer.add_scalar('Accuracy/train',args.train_accuracies[-1],epoch+1)

            if len(args.valid_accuracies) > 0:
                accuracies['valid'] = args.valid_accuracies[-1]
                args.writer.add_scalar('Accuracy/valid', args.valid_accuracies[-1], epoch + 1)

            if len(args.test_accuracies) > 0:
                accuracies['test'] = args.test_accuracies[-1]
                plotting_accuracies.append(args.test_accuracies[-1])
                args.writer.add_scalar('Accuracy/test', args.test_accuracies[-1], epoch + 1)



        if args.save_plot:

            plt.subplot(1, 2, 1)
            plt.title('Loss Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(args.train_losses, 'b')
            if args.__contains__('test_losses'):
                plt.plot(args.test_losses, 'r')
            if args.__contains__('valid_losses'):
                plt.plot(args.valid_losses, 'g')

            plt.subplot(1, 2, 2)
            plt.title('Accuracy Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Accuracy', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(args.train_accuracies, 'b')
            if args.__contains__('test_accuracies'):
                plt.plot(args.test_accuracies, 'r')
            if args.__contains__('valid_accuracies'):
                plt.plot(args.valid_accuracies, 'g')
            plt.savefig(os.path.join(args.log_path,'TrainingPlots'))
            plt.clf()
    args.writer.close()


    print('Training finished successfully. Model size: ', params,)
    if args.best_acc>0:
        print('Best acc: ', args.best_acc )


