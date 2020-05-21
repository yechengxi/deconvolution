'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

from .deconv import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, deconv=None):
        super(BasicBlock, self).__init__()
        if deconv:
            self.conv1 = deconv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.deconv = True
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.deconv = False


        self.shortcut = nn.Sequential()

        if not deconv:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            #self.bn1 = nn.GroupNorm(planes//16,planes)
            #self.bn2 = nn.GroupNorm(planes//16,planes)

            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                    #nn.GroupNorm(self.expansion * planes//16,self.expansion * planes)
                )
        else:
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    deconv(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
                )

    def forward(self, x):

        if self.deconv:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out

        else: #self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, deconv=None):
        super(Bottleneck, self).__init__()


        if deconv:
            self.deconv = True
            self.conv1 = deconv(in_planes, planes, kernel_size=1)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=stride, padding=1)
            self.conv3 = deconv(planes, self.expansion*planes, kernel_size=1)

        else:
            self.deconv = False

            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()

        if not deconv:
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        else:
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    deconv(in_planes, self.expansion * planes, kernel_size=1, stride=stride)
                )

    def forward(self, x):

        """
        No batch normalization for deconv.
        """
        if self.deconv:
            out = F.relu((self.conv1(x)))
            out = F.relu((self.conv2(out)))
            out = self.conv3(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, deconv=None,delinear=None,channel_deconv=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if deconv:
            self.deconv = True
            self.conv1 = deconv(3, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


        if not deconv:
            self.bn1 = nn.BatchNorm2d(64)

        #this line is really recent, take extreme care if the result is not good.
        if channel_deconv:
            self.deconv1=channel_deconv()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, deconv=deconv)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, deconv=deconv)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, deconv=deconv)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, deconv=deconv)
        if delinear:
            self.linear = delinear(512*block.expansion, num_classes)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, deconv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, deconv))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self,'bn1'):
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes,deconv,delinear,channel_deconv):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def ResNet34(num_classes,deconv,delinear,channel_deconv):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def ResNet50(num_classes,deconv,delinear,channel_deconv):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def ResNet101(num_classes,deconv,delinear,channel_deconv):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def ResNet152(num_classes,deconv,delinear,channel_deconv):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
