'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv import *


class Block(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, expansion, stride,deconv):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.shortcut = nn.Sequential()

        if deconv:
            self.deconv = True
            self.conv1=deconv(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=True)
            self.conv3 = deconv(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)

            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    deconv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
                    )
            # SE layers
            self.fc1 = deconv(out_planes, out_planes // 16, kernel_size=1)
            self.fc2 = deconv(out_planes // 16, out_planes, kernel_size=1)

        else:
            self.deconv = False
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, groups=planes, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1,
                              stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )

            # SE layers
            self.fc1 = nn.Conv2d(out_planes, out_planes//16, kernel_size=1)
            self.fc2 = nn.Conv2d(out_planes//16, out_planes, kernel_size=1)

    def forward(self, x):
        if self.deconv:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = self.conv3(out)

        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10, deconv=None,delinear=None,channel_deconv=None):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        if deconv:
            self.deconv=True
            self.conv1 = deconv(3, 32, kernel_size=3, stride=1, padding=1, bias=True,freeze=True,n_iter=10)
            
        else:
            self.deconv=False
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32,deconv=None)

        if channel_deconv:
            self.deconv1 = channel_deconv()

        if delinear:
            self.linear = delinear(cfg[-1][1], num_classes)
        else:
            self.linear = nn.Linear(cfg[-1][1], num_classes)

    def _make_layers(self, in_planes,deconv):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride,deconv))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self,'bn1'):
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)
        out = self.linear(out)
        return out


def EfficientNetB0(num_classes,deconv,delinear,channel_deconv):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 2),
           (6,  24, 2, 1),
           (6,  40, 2, 2),
           (6,  80, 3, 2),
           (6, 112, 3, 1),
           (6, 192, 4, 2),
           (6, 320, 1, 2)]
    return EfficientNet(cfg,num_classes,deconv,delinear,channel_deconv)


def test():
    net = EfficientNetB0()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


# test()
