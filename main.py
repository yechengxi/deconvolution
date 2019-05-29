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

    opts=parse_opts()

    from util import save_path_formatter
    log_dir=save_path_formatter(opts)
    opts.checkpoint_path=log_dir
    opts.result_path=log_dir
    opts.log_path=log_dir

    if opts.save_plot:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

    if opts.deconv:
        if opts.mode<5:
            opts.deconv=partial(DeConv2d,bias=opts.bias, eps=opts.eps, n_iter=opts.deconv_iter, mode=opts.mode, num_groups=opts.num_groups)
        elif opts.mode==5:
            opts.deconv = partial(FastDeconv, bias=opts.bias, eps=opts.eps, n_iter=opts.deconv_iter,num_groups=opts.num_groups)
    if opts.num_groups_final>0:
        opts.channel_deconv=partial(ChannelDeconv, num_groups=opts.num_groups_final,eps=opts.eps, n_iter=opts.deconv_iter)
    else:
        opts.channel_deconv=None

    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        torch.cuda.manual_seed(opts.seed)

    opts.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    if opts.dataset=='cifar10':
        opts.in_planes = 3
        opts.input_size=32
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
        opts.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        print("| Preparing CIFAR-10 dataset...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        opts.num_outputs = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif (opts.dataset == 'cifar100'):
        opts.in_planes = 3
        opts.input_size = 32
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
        opts.num_outputs = 100


    elif opts.dataset=='mnist':
        opts.in_planes=1
        opts.input_size = 28
        trainset= torchvision.datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        testset=torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        opts.num_outputs = 10
    elif opts.dataset == 'fashion':
        opts.in_planes = 1
        opts.input_size = 28
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                              transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
        opts.num_outputs = 10
    elif opts.dataset=='stl10':
        opts.in_planes = 3
        opts.input_size = 96
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
        opts.classes=('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

        print("| Preparing STL10 dataset...")

        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=True, transform=transform_test)
        opts.num_outputs = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif opts.dataset=='svhn':
        opts.in_planes = 3
        opts.input_size = 32
        trainset = torchvision.datasets.SVHN(root='./data',  split='train', download=True,
                                              transform=transforms.Compose([
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.3782,  0.3839,  0.4100),(0.1873,  0.1905,  0.1880))
                                              ]))
        #from util import *
        #mean,std=get_mean_and_std(trainset)
        #print(mean,std)
        testset = torchvision.datasets.SVHN(root='./data',  split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3782, 0.3839, 0.4100), (0.1873, 0.1905, 0.1880))
        ]))
        opts.num_outputs = 10
    elif (opts.dataset == 'imagenet16') or (opts.dataset == 'imagenet32'):
        opts.in_planes = 3
        if (opts.dataset == 'imagenet16'):
            opts.input_size = 16
            datapath='./data/Imagenet16'
        if (opts.dataset == 'imagenet32'):
            opts.input_size = 32
            datapath = './data/Imagenet32'

        transform_train = transforms.Compose([
            transforms.RandomCrop(opts.input_size, padding=int(math.log2(opts.input_size)-1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opts.batch_size, shuffle=True,
                                              num_workers=opts.num_workers)


    if not os.path.exists(opts.log_path):
        os.makedirs(opts.log_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                                  ['epoch','loss', 'top1', 'top5','time'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                     ['epoch','batch', 'loss', 'top1', 'top5', 'time'])
    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
                                    ['epoch', 'loss', 'top1', 'top5', 'time'])

    # Model

    if opts.tensorboardX:
        from tensorboardX import SummaryWriter
        opts.writer = SummaryWriter(opts.log_path)
    elif opts.tensorboard:
        from tf_logger import Logger
        opts.writer=Logger(opts.log_path)

    print('==> Building model..')

    if opts.deconv:
        opts.batchnorm=False
        print('************ Batch norm disabled when deconv is used. ************')

    if (not opts.deconv) and opts.channel_deconv:
        print('************ Channel Deconv is used on the original model, this accelrates the training. If you want to turn it off set --num-groups-final 0 ************')

    if opts.arch =='LeNet':
        if opts.deconv:
            net = LeNetDeconv()

    if opts.arch == 'vgg19':
        net = VGG('VGG19',num_classes=opts.num_outputs, deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch == 'vgg13':
        net = VGG('VGG13',num_classes=opts.num_outputs, deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch == 'vgg11':
        net = VGG('VGG11',num_classes=opts.num_outputs, deconv=opts.deconv,channel_deconv=opts.channel_deconv, dropout=opts.dropout_rate)

    if opts.arch == 'vgg11d':
        from models.vgg_imagenet import vgg11d
        net = vgg11d('VGG11d',num_classes=opts.num_outputs, deconv=opts.deconv,channel_deconv=opts.channel_deconv)


    if opts.arch=='resnet':
        net = ResNet18(num_classes=opts.num_outputs,deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch=='resnet18d':
        from models.resnet_imagenet import resnet18d
        net = resnet18d(num_classes=opts.num_outputs,deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch=='resnet34':
        net = ResNet34(num_classes=opts.num_outputs,deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch=='resnet50':
        net = ResNet50(num_classes=opts.num_outputs,deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    # net = GoogLeNet()
    if opts.arch == 'densenet':
        net = densenet_cifar()

    if opts.arch == 'densenet121':
        net = DenseNet121(num_classes=opts.num_outputs,deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch == 'densenet121d':
        from models.densenet_imagenet import densenet121d
        net = densenet121d(num_classes=opts.num_outputs,deconv=opts.deconv,channel_deconv=opts.channel_deconv)

    if opts.arch == 'simple_v1':
        from models.simple import *
        net = SimpleCNN_v1(channels_in=opts.in_planes,kernel_size=opts.input_size,num_outputs=opts.num_outputs,method=opts.method)

    if opts.arch == 'simple_v2':
        from models.simple import *
        net = SimpleCNN_v2(channels_in=opts.in_planes,kernel_size=3,hidden_layers=10,hidden_channels=4,num_outputs=opts.num_outputs,method=opts.method)

    if opts.arch == 'mlp':
        from models.simple import *
        net = MLP(input_nodes=784, hidden_nodes=128,hidden_layers=3,method=opts.method,num_outputs=opts.num_outputs)

    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()



    if opts.loss=='CE':
        opts.criterion = nn.CrossEntropyLoss()
        if opts.use_gpu:
            opts.criterion = nn.CrossEntropyLoss().cuda()
            #opts.criterion = torch.nn.DataParallel(opts.criterion)
    elif opts.loss=='L2':
        opts.criterion = nn.MSELoss()
        if opts.use_gpu:
            opts.criterion = nn.MSELoss().cuda()
            # opts.criterion = torch.nn.DataParallel(opts.criterion)


    opts.logger_n_iter = 0


    # Training


    print(opts)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params,'trainable parameters in the network.')

    set_parameters(opts)



    lr = opts.lr

    
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    if opts.optimizer == 'SGD':
        opts.current_optimizer = optim.SGD(parameters, lr=lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.optimizer == 'Adam':
        opts.current_optimizer = optim.Adam(parameters, lr=lr, weight_decay=opts.weight_decay)
        #from my_optim import Adam3
        #opts.current_optimizer = Adam3(parameters, lr=lr, weight_decay=opts.weight_decay)

    if opts.lr_scheduler=='multistep':

        milestones=[int(opts.milestone*opts.epochs)]
        while milestones[-1]+milestones[0]<opts.epochs:
            milestones.append(milestones[-1]+milestones[0])

        opts.current_scheduler = optim.lr_scheduler.MultiStepLR(opts.current_optimizer, milestones=milestones, gamma=opts.multistep_gamma)

    if opts.lr_scheduler=='cosine':
        total_steps = math.ceil(len(trainset)/opts.batch_size)*opts.epochs
        opts.current_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opts.current_optimizer, total_steps, eta_min=0, last_epoch=-1)

    opts.total_steps = math.ceil(len(trainset)/opts.batch_size)*opts.epochs
    opts.cur_steps=0

    if opts.use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    device = torch.device("cuda")

    plotting_accuracies = []

    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            opts.best_acc = checkpoint['best_acc']
            if hasattr(net,'module'):
                net.module.load_state_dict(checkpoint['state_dict'])
            else:
                net.load_state_dict(checkpoint['state_dict'])
            opts.current_optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))


    if opts.resume:
        lr = opts.lr
        for param_group in opts.current_optimizer.param_groups:
            param_group['lr'] = lr
        if opts.lr_scheduler == 'multistep':
            for i in range(opts.start_epoch):
                opts.current_scheduler.step()
        if opts.lr_scheduler == 'cosine':
            total_steps = math.ceil(len(trainset) / opts.batch_size) * opts.start_epoch
            for i in range(total_steps):
                opts.current_scheduler.step()



    for epoch in range(opts.start_epoch, opts.start_epoch + opts.epochs):
        opts.epoch = epoch
        if opts.lr_scheduler == 'multistep':
            opts.current_scheduler.step()
        if opts.lr_scheduler == 'multistep' or opts.lr_scheduler=='cosine':
            print('Current learning rate:', opts.current_scheduler.get_lr()[0])



        opts.data_loader = train_loader
        train_net(net,opts)
        opts.data_loader = test_loader

        opts.validating=False
        opts.testing=True

        eval_net(net,opts)

        if opts.tensorboard or opts.tensorboardX:
            #Log scalar values (scalar summary)
            losses={'train':opts.train_losses[-1]}
            if opts.tensorboardX:
                opts.writer.add_scalar('Loss/train', opts.train_losses[-1], epoch + 1)
            else:
                opts.writer.scalar_summary('Loss/train',opts.train_losses[-1],epoch+1)

            if len(opts.valid_losses) > 0:
                losses['valid'] = opts.valid_losses[-1]

                if opts.tensorboardX:
                    opts.writer.add_scalar('Loss/valid', opts.valid_losses[-1], epoch + 1)
                else:
                    opts.writer.scalar_summary('Loss/valid', opts.valid_losses[-1], epoch + 1)

            if len(opts.test_losses) > 0:
                losses['test'] = opts.test_losses[-1]

                if opts.tensorboardX:
                    opts.writer.add_scalar('Loss/test', opts.test_losses[-1], epoch + 1)
                else:
                    opts.writer.scalar_summary('Loss/test', opts.test_losses[-1], epoch + 1)

            accuracies={'train':opts.train_accuracies[-1]}

            if opts.tensorboardX:
                opts.writer.add_scalar('Accuracy/train',opts.train_accuracies[-1],epoch+1)
            else:
                opts.writer.scalar_summary('Accuracy/train',opts.train_accuracies[-1],epoch+1)

            if len(opts.valid_accuracies) > 0:
                accuracies['valid'] = opts.valid_accuracies[-1]
                if opts.tensorboardX:
                    opts.writer.add_scalar('Accuracy/valid', opts.valid_accuracies[-1], epoch + 1)
                else:
                    opts.writer.scalar_summary('Accuracy/valid', opts.valid_accuracies[-1], epoch + 1)

            if len(opts.test_accuracies) > 0:
                accuracies['test'] = opts.test_accuracies[-1]
                plotting_accuracies.append(opts.test_accuracies[-1])
                if opts.tensorboardX:
                    opts.writer.add_scalar('Accuracy/test', opts.test_accuracies[-1], epoch + 1)
                else:
                    opts.writer.scalar_summary('Accuracy/test', opts.test_accuracies[-1], epoch + 1)

        if opts.save_plot:

            plt.subplot(1, 2, 1)
            plt.title('Loss Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(opts.train_losses, 'b')
            if opts.__contains__('test_losses'):
                plt.plot(opts.test_losses, 'r')
            if opts.__contains__('valid_losses'):
                plt.plot(opts.valid_losses, 'g')

            plt.subplot(1, 2, 2)
            plt.title('Accuracy Plot', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel('Accuracy', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.plot(opts.train_accuracies, 'b')
            if opts.__contains__('test_accuracies'):
                plt.plot(opts.test_accuracies, 'r')
            if opts.__contains__('valid_accuracies'):
                plt.plot(opts.valid_accuracies, 'g')
            plt.savefig(os.path.join(opts.log_path,'TrainingPlots'))
            plt.clf()



    
    RESULT_TEST_ACCURACIES = opts.test_accuracies

    with open("test_accuracies", "a+") as writeFile:
        wr = csv.writer(writeFile, delimiter=',')
        wr.writerow(RESULT_TEST_ACCURACIES)

    
        
    RESULT_TRAIN_ACCURACIES = opts.train_accuracies

    with open("train_accuracies", "a+") as writeFile:
        wr = csv.writer(writeFile, delimiter=',')
        wr.writerow(RESULT_TRAIN_ACCURACIES)

        
        
    


    print('Training finished successfully. Model size: ', params,)
    if opts.best_acc>0:
        print('Best acc: ', opts.best_acc )


