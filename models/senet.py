"""SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1,deconv=None):
        super(BasicBlock, self).__init__()
        self.shortcut = nn.Sequential()

        if not deconv:

            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )

            # SE layers
            self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
            self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

        else:

            self.conv1 = deconv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

            if stride != 1 or in_planes != planes:
                self.shortcut = deconv(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            # SE layers
            self.fc1 = deconv(planes, planes // 16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
            self.fc2 = deconv(planes // 16, planes, kernel_size=1)


    def forward(self, x):
        if self.deconv:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1,deconv=None):
        super(PreActBlock, self).__init__()
        if not deconv:
            self.deconv=False
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
                )

            # SE layers
            self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
            self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

        else:
            self.deconv = True
            self.conv1 = deconv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

            if stride != 1 or in_planes != planes:
                self.shortcut = deconv(in_planes, planes, kernel_size=1, stride=stride, bias=True)

            # SE layers
            self.fc1 = deconv(planes, planes // 16, kernel_size=1)
            self.fc2 = deconv(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        if self.deconv:
            out = F.relu(x)
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(out))
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, deconv=None,delinear=None,channel_deconv=None):
        super(SENet, self).__init__()
        self.in_planes = 64

        if deconv is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = deconv(3, 64, kernel_size=3, stride=1, padding=1, bias=True,freeze=True,n_iter=15)

        if channel_deconv:
            self.deconv1=channel_deconv()

        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1, deconv=deconv)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, deconv=deconv)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, deconv=deconv)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, deconv=deconv)
        
        if delinear:
            self.linear = delinear(512, num_classes)
        else:
            self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, deconv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, deconv))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        if hasattr(self, 'bn1'):
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


def SENet18(num_classes,deconv,delinear,channel_deconv):
    return SENet(PreActBlock, [2,2,2,2],num_classes,deconv,delinear,channel_deconv)


def test():
    net = SENet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
