 # Network Deconvolution

Convolution is a central operation in Convolutional Neural Networks (CNNs), which applies a kernel to overlapping regions shifted across the image. However, because of the strong  correlations in real-world image data, convolutional kernels are in effect re-learning redundant data. In this work, we show that this redundancy has made neural network training challenging, and propose network deconvolution, a procedure which optimally removes pixel-wise and channel-wise correlations before the data is fed into each layer. Network deconvolution can be efficiently calculated at a fraction of the computational cost of a convolution layer. We also show that the deconvolution filters in the first layer of the network resemble the center-surround structure found in biological neurons in the visual regions of the brain. Filtering with such kernels results in a sparse representation, a desired property that has been missing in the training of neural networks. Learning from the sparse representation promotes faster convergence and superior results without the use of batch normalization. We apply our network deconvolution operation to 10 modern neural network models by replacing batch normalization within each. Extensive experiments show that the network deconvolution operation is able to deliver performance improvement in all cases on the CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, Cityscapes, and ImageNet datasets.  

>@inproceedings{  
>Ye2020Network,  
>title={Network Deconvolution},  
>author={Chengxi Ye and Matthew Evanusa and Hua He and Anton Mitrokhin and Tom Goldstein and James A. Yorke and Cornelia Fermuller and Yiannis Aloimonos},  
>booktitle={International Conference on Learning Representations},  
>year={2020},  
>url={https://openreview.net/forum?id=rkeu30EtvS }  
>}  



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
 



   
 ## 2. ImageNet dataset:


1. original resnet18 (90 epochs, use --epochs xx to change)
```
python main_imagenet.py -a resnet18 -j 32 imagenet/ILSVRC/Data/CLS-LOC 
```
2. deconv resnet18
```
python main_imagenet.py -a resnet18d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True
```

 
 ## 3. Semantic segmentation:
 
 Go to the Segmentation folder and follow the instructions in the ReadMe file.  

