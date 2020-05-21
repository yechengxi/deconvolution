'''PNASNet in PyTorch.

Paper: Progressive Neural Architecture Search
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride,deconv=None):
        super(SepConv, self).__init__()
        if not deconv:
            self.conv1 = nn.Conv2d(in_planes, out_planes,
                                   kernel_size, stride,
                                   padding=(kernel_size-1)//2,
                                   bias=False, groups=in_planes)
            self.bn1 = nn.BatchNorm2d(out_planes)
        else:
            self.conv1 = deconv(in_planes, out_planes,
                                   kernel_size, stride,
                                   padding=(kernel_size - 1) // 2,
                                   bias=True, groups=in_planes)

    def forward(self, x):
        if hasattr(self, 'bn1'):
            return self.bn1(self.conv1(x))
        else:
            return self.conv1(x)


class CellA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1,deconv=None):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride,deconv=deconv)
        if stride==2:
            if not deconv:
                self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(out_planes)
            else:
                self.conv1 = deconv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            if hasattr(self, 'bn1'):
                y2 = self.bn1(self.conv1(y2))
            else:
                y2 = self.conv1(y2)

        return F.relu(y1+y2)

class CellB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1,deconv=None):
        super(CellB, self).__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride,deconv=deconv)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride,deconv=deconv)
        # Right branch
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride,deconv=deconv)
        if stride==2:
            if not deconv:
                self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(out_planes)
            else:
                self.conv1 = deconv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)

        # Reduce channels
        if not deconv:
            self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
        else:
            self.conv2 = deconv(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            if hasattr(self, 'bn1'):
                y3 = self.bn1(self.conv1(y3))
            else:
                y3 = self.conv1(y3)

        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = F.relu(y1+y2)
        b2 = F.relu(y3+y4)
        y = torch.cat([b1,b2], 1)
        if hasattr(self, 'bn2'):
            return F.relu(self.bn2(self.conv2(y)))
        else:
            return F.relu(self.conv2(y))


class PNASNet(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes, num_classes=10, deconv=None,delinear=None,channel_deconv=None):
        super(PNASNet, self).__init__()
        self.in_planes = num_planes
        self.cell_type = cell_type

        if deconv is None:
            self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_planes)
        else:
            self.conv1 = deconv(3, num_planes, kernel_size=3, stride=1, padding=1, bias=True, freeze=True, n_iter=10)

        if channel_deconv:
            self.deconv1 = channel_deconv()


        self.layer1 = self._make_layer(num_planes, num_cells=6,deconv=deconv)
        self.layer2 = self._downsample(num_planes*2,deconv=deconv)
        self.layer3 = self._make_layer(num_planes*2, num_cells=6,deconv=deconv)
        self.layer4 = self._downsample(num_planes*4,deconv=deconv)
        self.layer5 = self._make_layer(num_planes*4, num_cells=6,deconv=deconv)

        if delinear:
            self.linear = delinear(num_planes*4, num_classes)
        else:
            self.linear = nn.Linear(num_planes*4, num_classes)

    def _make_layer(self, planes, num_cells,deconv):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1,deconv=deconv))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes,deconv):
        layer = self.cell_type(self.in_planes, planes, stride=2,deconv=deconv)
        self.in_planes = planes
        return layer

    def forward(self, x):
        if hasattr(self,'bn1'):
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
        return out


def PNASNetA(num_classes,deconv,delinear,channel_deconv):
    return PNASNet(CellA, num_cells=6, num_planes=44,num_classes=num_classes,deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)

def PNASNetB(num_classes,deconv,delinear,channel_deconv):
    return PNASNet(CellB, num_cells=6, num_planes=32,num_classes=num_classes,deconv=deconv,delinear=delinear,channel_deconv=channel_deconv)


def test():
    net = PNASNetB()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
