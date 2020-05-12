# Semantic Segmentation

### Training from scratch on the Cityscapes dataset 


#### DeepLabv3:  

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cityscapes --model deeplabv3_resnet50 -b 8 --epochs 30 --deconv False --pretrained-backbone False --lr 0.1 &  

CUDA_VISIBLE_DEVICES=1 python train.py --dataset cityscapes --model deeplabv3_resnet50d -b 8 --epochs 30  --deconv True --pretrained-backbone False --lr 0.1 &

#### FCN:  

CUDA_VISIBLE_DEVICES=2 python train.py --dataset cityscapes --model fcn_resnet50 -b 8 --epochs 30 --deconv False --pretrained-backbone False --lr 0.1 &  

CUDA_VISIBLE_DEVICES=3 python train.py --dataset cityscapes --model fcn_resnet50d -b 8 --epochs 30  --deconv True --pretrained-backbone False --lr 0.1 &

