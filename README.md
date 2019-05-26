 # Deconv

<<<<<<< HEAD
 ## Install Dependencies

 We recommend using pip to install the required dependencies:

 '''
	
  
 '''

 ## Settings Overview
 We have included a few settings you can add into the run command.

 The basic run command (for non-imagenet dataset) is:

 '''
 python main.py --[keyword] [argument] ...
 '''

 The major keywords to note are:

 deconv - set to True or False if you want to test deconv (True) or BN (False)
 arch - use a given architecture (resnet50, vgg11, vgg13, vgg19, densenet121)
 wd - sets the weight decay to a given value
 batch-size - sets the batch size
 method - sets the method for deconv.  3 for channel-only and 4 for full
 epochs - the number of epochs to run
 dataset - the dataset to use (cifar10, cifar100) (for imagenet you need the other main file)
 lr - sets the learning rate


 ## 1. For logistic regression (--loss L2 for L2 linear regression): 
=======
 # 1. For logistic regression (--loss L2 for L2 linear regression): 
>>>>>>> 13473f9b141a3563a8ba2423f630a085f79b8724

method 1: regular conv, method 2: channel-wise deconv + conv, method 3: channel and pixel deconv + conv 4: group deconv (rearrange pixels into groups then deconv) 

python main.py --lr .1 --optimizer SGD --arch simple_v1 --epochs 1 --dataset cifar10  --batch-size 512 --msg True --method 1 --loss CE
 
<<<<<<< HEAD
 ## 2. For a simple cnn: 
=======
 # 2. For a 3-hidden-layer fully connected network: 
>>>>>>> 13473f9b141a3563a8ba2423f630a085f79b8724

method 1: vanilla network + sgd, method 2: batch norm + sgd, method 3: channel deconv with 32 groups 

python main.py --lr .1 --optimizer SGD --arch mlp --epochs 20 --dataset mnist  --batch-size 512 

 ## 3. Train a resnet on CIFAR-10 (with/without deconv):

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch resnet --epochs 100 --dataset cifar10  --batch-size 512 --msg True --deconv False --num-groups-final 0 >resnet18.cifar10.100ep.log&

(deconv:)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --lr .1 --optimizer SGD --arch resnet --epochs 100 --dataset cifar10  --batch-size 512 --msg True --deconv True >resnet18d.cifar10.100ep.log&

(fast:)

python main.py --lr .1 --optimizer SGD --arch resnet --epochs 20 --dataset cifar10  --batch-size 128 --msg True --deconv True --wd 1e-3 >resnet18d.cifar10.20ep.log&


 # 4. Train a resnet on CIFAR-100 (with/without deconv):

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch resnet50 --epochs 100 --dataset cifar100  --batch-size 512 --msg True --deconv False --num-groups-final 0 >resnet50.cifar100.100ep.log&

(74.540%)

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch resnet50 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv False --num-groups-final 0 >resnet50.cifar100.100ep.bs128.log&

(78.79%)

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv False --num-groups-final 0 >vgg11.cifar100.100ep.bs128.log&

(70.550%)

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 100 --dataset cifar100  --batch-size 512 --msg True --deconv False --num-groups-final 0 >vgg.cifar100.100ep.bs512.log&

(69.390%)
 
(deconv:)

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv True >vgg_d.cifar100.100ep.bs128.log&

(72.34%)

(fast:)

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 20 --dataset cifar100  --batch-size 128 --deconv True --wd 1e-3 >vgg_d.cifar100.20ep.bs128.wd.1e-3.log&

(70.7%)

CUDA_VISIBLE_DEVICES=0,1 python main.py --lr .2 --optimizer SGD --arch vgg11 --epochs 20 --dataset cifar100  --batch-size 512 --deconv True --wd 1e-3 >vgg_d.cifar100.20ep.bs512.wd.1e-3.lr.2.log&

(69.650%)

 # 5. imagenet dataset:


1. original resnet18 (90 epochs, use --epochs xx to change)

python main_imagenet.py -a resnet18 -j 32 imagenet/ILSVRC/Data/CLS-LOC >resnet18.log &

2. deconv resnet18

python main_imagenet.py -a resnet18d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True >resnet18d.log &

3. deconv vgg11
 
<<<<<<< HEAD
python main_imagenet.py -a vgg11d -j 32 /vulcan/scratch/cxy/Data/imagenet/ILSVRC/Data/CLS-LOC --deconv True --lr 0.01 >vgg11d.imagenet.log &

###TODO:
vs vgg11_bn

 
 ## important tricks

On the vulcan server, run:

CUDA_VISIBLE_DEVICES=0 tensorboard --logdir=checkpoints --port=6006 2>tb.tmp&

Then on your local machine run:

ssh [username]@openlab.umiacs.umd.edu -L 0.0.0.0:16006:vulcan[XX].umiacs.umd.edu:6006

Then open: localhost:6006 on your local machine.
=======
python main_imagenet.py -a vgg11d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True --lr 0.01 >vgg11d.imagenet.log &
>>>>>>> 13473f9b141a3563a8ba2423f630a085f79b8724

 


