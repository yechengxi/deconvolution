 # Network Deconvolution

 ## FAQ
 ### Why is it called deconvolution?

Imagine you have a sharp signal I. However, nature is applying a blur to the signal so what you observe is: B=K*I. This blurred signal makes machine learning very hard.

What we want to do is a deconvolution that recovers the clear/white signal I=inv(K)*B. And we use this deconvolved signal for kernel learning in the CNN. The blur calculated from the blurred observation using the square root of the covariance. 

 ### What is the walltime?
 
On ImageNet dataset, we benchmark the walltime with 1 GTX1080Ti GPU and 16 CPUs. It takes 1 hour/epoch to train a regular ResNet-18. It takes 1 hour and 3 minutes to train the same network with network deconvolution.

In the latest update, we introduce a sampling trick for acceleration. Note that the covariance matrix is usually small compared to the number of pixels involved in a batch of data. By default we use a 3x subsampling when calculating the covariance matrix. Experimentally we notice this does not reduce the quality. Instead, it allows us to split data over fewer GPUs and improve the quality.  

 ## Environment

 Our code was developed and requires Ubuntu 14 or greater, and python 3.5 or greater.

 You can install python3.5 in ubuntu using:

```
 sudo add-apt-repository ppa:deadsnakes/ppa
 sudo apt-get update
 sudo apt-get install python3.5

```
 ## Install Dependencies

 This code requires the use of python3.5 or greater.
 
 We recommend using pip to install the required dependencies.  For python3+, use pip3:

 Install scipy and numpy:
 ```
 sudo pip3 install scipy numpy tensorboardX matplotlib
 ```

 Install PyTorch:
 ```
 sudo pip3 install torch torchvision
  
 ```
 
 (optional, for visualization) Install tensorflow:

 ```
 sudo pip3 install tensorflow
 
 ```



## Running in Python

 If you have python 2.7 installed as your default, some machines require you to call the newer versions of python with a separate name.  For example, you may have to run:

 ```
 python3.5 example_code.py
 ```

 to actually invoke python3.5 on your machine. If this is the case, wherever we have written "python main.py" you will just have to replace with "python3.5 main.py" or whichever version you have installed. 

 ## Checking your GPU

 This code also requires at least one CUDA-enabled GPU from NVIDIA to run properly. You can test to see if you have a CUDA GPU available by running:

 ```
 nvidia-smi
 ```

 in your terminal.
 
 ## Settings Overview
 We have included a few settings you can add into the run command.

 The basic run command (for non-imagenet dataset) is:

 ```
 python main.py --[keyword1] [argument1] --[keyword2] [argument2]  ...
 ```

 The major keywords to note are:

 * deconv - set to True or False if you want to test deconv (True) or BN (False)
 * arch - use a given architecture (resnet50, vgg11, vgg13, vgg19, densenet121)
 * wd - sets the weight decay to a given value
 * batch-size - sets the batch size
 * mode - sets the method for deconv.  3 for channel-only and 4 for full, 5 for a fast and efficient approximate of 4 (default)
 * epochs - the number of epochs to run
 * dataset - the dataset to use (cifar10, cifar100) (for imagenet you need the other main file)
 * lr - sets the learning rate
 * num groups final - additional channel deconvolution done at the last fully connected layer.  Set to 0 for Batch norm and 512 for deconvolution

 ## Running the examples from the paper

 We ran all of our experiments on two to four Nvidia Geoforce 1080Ti GPUs (for
 CIFAR-10 and CIFAR-100), but your needs may vary. 

 Depending on how many GPUs you have, you will need to change the CUDA_VISIBLE_DEVICES parameter before 'python main.py'.

run:

```
nvidia-smi
```
and see how many GPUS you have.  They will be given a number from 0 -> n, where n is the number of GPUs.  Wherever we have listed CUDA_VISIBLE_DEVICES, put 0,->n to the right of the equal sign.  For example, if you see three GPUs, numbered 0,1,2, then you would put

```
CUDA_VISIBLE_DEVICES=0,1,2
```
Before python main.py. Putting less devices will still work, but it will be slower. To be safe, you can always just copy the code we have given you here, which sets CUDA_VISIBLE_DEVICES to be 0, the first GPU.

 As an example, to run our settings for the CIFAR-10 20-epoch run, with .001 weight decay and 128 batch size, on the vgg11 architecture, you would run:

```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 20 --dataset cifar10  --batch-size 128 --msg True --deconv False --num-groups-final 0 --wd .001
```
for batch norm, and

```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 20 --dataset cifar10  --batch-size 128 --msg True --deconv True --num-groups-final 512 --wd .001
```

for deconvolution
 
 ## 1. For logistic regression (--loss L2 for L2 linear regression): 

```
python main.py --lr .1 --optimizer SGD --arch simple_v1 --epochs 1 --dataset cifar10  --batch-size 512 --msg True --method 1 --loss CE
 

```


## 2. Train a resnet on CIFAR-10 (with/without deconv):

batch normalization:
```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch resnet --epochs 100 --dataset cifar10  --batch-size 512 --msg True --deconv False --num-groups-final 0 
```
deconv:
```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch resnet --epochs 100 --dataset cifar10  --num-groups-final 512 --batch-size 512 --msg True --deconv True 
```

 ## 3. Train a resnet50 on CIFAR-100 (with/without deconv):

batch norm.:
```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch resnet50 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv False --num-groups-final 0 
```
deconv.:
```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch resnet50 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv True --num-groups-final 512 
```
## 4. Train a vgg[11/13/19] network on CIFAR-10/100 (with/without deconv)

For vgg13 or vgg19, replace vgg11 with either vgg13 or vgg19. 

for batch norm:
```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv False --num-groups-final 0
```
deconv:

```

CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv True --num-groups-final 512

```
# 5. Training with Channel Deconv Only

To train a network with channel deconv only, use mode '3' in the --mode argument with --deconv set to True.

example (running vgg11 with channel deconv only):

```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 100 --dataset cifar100  --batch-size 128 --msg True --deconv True --mode 3 --num-groups-final 512
```
   
 # 6. imagenet dataset:


1. original resnet18 (90 epochs, use --epochs xx to change)
```
python main_imagenet.py -a resnet18 -j 32 imagenet/ILSVRC/Data/CLS-LOC 
```
2. deconv resnet18
```
python main_imagenet.py -a resnet18d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True
```
3. deconv vgg11
 
```
python main_imagenet.py -a vgg11d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True --lr 0.01 
```

 


