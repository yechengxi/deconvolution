"""DenseNet in PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate,deconv):
        super(Bottleneck, self).__init__()
        if not deconv:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(4*growth_rate)
            self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = deconv(in_planes, 4 * growth_rate, kernel_size=1)
            self.conv2 = deconv(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
    def forward(self, x):
        if hasattr(self,'bn1'):
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
        else:
            out = self.conv1(F.relu(x))
            out = self.conv2(F.relu(out))
        out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes,deconv):
        super(Transition, self).__init__()
        if not deconv:
            self.bn = nn.BatchNorm2d(in_planes)
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        else:
            self.conv=deconv(in_planes, out_planes, kernel_size=1)
    def forward(self, x):
        if hasattr(self,'bn'):
            out = self.conv(F.relu(self.bn(x)))
        else:
            out = self.conv(F.relu(x))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10,deconv=None,delinear=None,channel_deconv=None):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        if not deconv:
            self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = deconv(3, num_planes, kernel_size=3, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0],deconv)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes,deconv)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1],deconv)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes,deconv)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2],deconv)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes,deconv)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3],deconv)
        num_planes += nblocks[3]*growth_rate
        if not deconv:
            self.bn = nn.BatchNorm2d(num_planes)
        elif channel_deconv:
            self.channel_deconv=channel_deconv()
        
        if delinear:
            self.linear = delinear(num_planes, num_classes)
        else:
            self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock,deconv):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate,deconv))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        if hasattr(self,'bn'):
            out = self.bn(out)
        
        out=F.relu(out)

        if hasattr(self, 'channel_deconv'):
            out = self.channel_deconv(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(num_classes,deconv,delinear,channel_deconv):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def DenseNet169(num_classes,deconv,delinear,channel_deconv):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32,num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def DenseNet201(num_classes,deconv,delinear,channel_deconv):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32,num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def DenseNet161(num_classes,deconv,delinear,channel_deconv):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48,num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def densenet_cifar(num_classes,deconv,delinear,channel_deconv):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def test_densenet():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)

# test_densenet()
