 # Network Deconvolution

>@inproceedings{  
>Ye2020Network,  
>title={Network Deconvolution},  
>author={Chengxi Ye and Matthew Evanusa and Hua He and Anton Mitrokhin and Tom Goldstein and James A. Yorke and Cornelia Fermuller and Yiannis Aloimonos},  
>booktitle={International Conference on Learning Representations},  
>year={2020},  
>url={https://openreview.net/forum?id=rkeu30EtvS}  
>}  

 ## FAQ
 ### Why is it called deconvolution?

Imagine you have a sharp signal I. However, nature is applying a blur to the signal so what you observe is: B=K*I. This blurred signal makes machine learning very hard.

What we want to do is a deconvolution that recovers the clear/white signal I=inv(K)*B. And we use this deconvolved signal for kernel learning in the CNN. The blur calculated from the blurred observation using the square root of the covariance. 

 ## Install Dependencies

 This code requires the use of python3.5 or greater.
 
 We recommend using pip to install the required dependencies.

 ```
 pip install scipy numpy tensorboard matplotlib
 ```

 Install PyTorch:
 ```
 pip install torch torchvision
  
 ```
 
 (optional, for visualization) Install tensorflow:

 ```
 pip3 install tensorflow
 
 ```


 
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
 * epochs - the number of epochs to run
 * dataset - the dataset to use (cifar10, cifar100) (for imagenet you need the other main file)
 * lr - sets the learning rate
 * block - block size in deconvolution
 * block-fc - block size in decorrelating the fully connected layers.


 ## 1. Running the examples from the paper

 
As an example, to run our settings for the CIFAR-10 20-epoch run, with .001 weight decay and 128 batch size, on the vgg11 architecture, you would run:

```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 20 --dataset cifar10  --batch-size 128 --msg True --deconv False --block-fc 0 --wd .001
```
for batch norm, and

```
CUDA_VISIBLE_DEVICES=0 python main.py --lr .1 --optimizer SGD --arch vgg11 --epochs 20 --dataset cifar10  --batch-size 128 --msg True --deconv True --block-fc 512 --wd .001
```

for deconvolution
 



   
 # 2. imagenet dataset:


1. original resnet18 (90 epochs, use --epochs xx to change)
```
python main_imagenet.py -a resnet18 -j 32 imagenet/ILSVRC/Data/CLS-LOC 
```
2. deconv resnet18
```
python main_imagenet.py -a resnet18d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True
```

 


