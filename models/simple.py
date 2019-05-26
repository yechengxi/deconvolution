import torch
import torch.nn as nn
import torch.nn.functional as F

from .deconv import *

class SimpleCNN_v1(nn.Module):
    def __init__(self, channels_in=3,kernel_size=32,num_outputs=10,method=2):
        super(SimpleCNN_v1, self).__init__()
        if method==1:
            self.conv = nn.Conv2d(channels_in, num_outputs, kernel_size)

        elif method==2:
            self.conv = nn.Sequential(ChannelDeconv(num_groups=3),nn.Conv2d(channels_in, num_outputs, kernel_size))

        elif method==3:
            self.conv=DeConv2d(channels_in,num_outputs,kernel_size)
        elif method==4:
            #1<= num_groups <= kernel_size*kernel_size*channels_in
            num_groups=32
            self.deconv = ChannelDeconv(num_groups=num_groups)
            self.conv=nn.Conv2d(channels_in, num_outputs, kernel_size)

        self.method=method


    def forward(self, x):
        N, C, H, W = x.shape

        if self.method==4:
            x=self.deconv(x.view(N,-1,1,1)).view(N,C,H,W)

        out=self.conv(x).view(x.shape[0],-1)

        return out


class SimpleCNN_v2(nn.Module):
    def __init__(self, channels_in=3,kernel_size=3,num_outputs=10,hidden_channels=4,hidden_layers=10,method=2):
        super(SimpleCNN_v2, self).__init__()

        self.method=method

        self.layers=nn.ModuleList()
        if method==1:
            self.layers.append(nn.Sequential(nn.Conv2d(channels_in,hidden_channels,kernel_size),nn.BatchNorm2d(hidden_channels),nn.ReLU()))
            padding = 0
            for i in range(hidden_layers):
                if i>=12:
                    padding=1
                self.layers.append(nn.Sequential(nn.Conv2d(hidden_channels,hidden_channels+4,kernel_size,padding=padding),nn.BatchNorm2d(hidden_channels+4),nn.ReLU()))
                hidden_channels=hidden_channels+4
            self.layers.append(nn.Sequential(nn.Conv2d(hidden_channels,num_outputs,kernel_size),nn.BatchNorm2d(num_outputs)))
        elif method==2:
            num_groups=16
            self.layers.append(nn.Sequential(DeConv2d(channels_in,hidden_channels,kernel_size,mode=3,num_groups=min(num_groups,channels_in)),nn.ReLU()))
            padding = 0
            for i in range(hidden_layers):
                if i>=12:
                    padding=1
                self.layers.append(nn.Sequential(DeConv2d(hidden_channels,hidden_channels+4,kernel_size,padding=padding,mode=3,num_groups=min(num_groups,hidden_channels)),nn.ReLU()))
                hidden_channels=hidden_channels+4
            self.layers.append(nn.Sequential(DeConv2d(hidden_channels,num_outputs,kernel_size,mode=3,num_groups=min(num_groups,hidden_channels))))

            """
            self.layers.append(nn.Sequential(ChannelDeconv(channels_in),nn.Conv2d(channels_in,hidden_channels,kernel_size),nn.ReLU()))
            padding = 0
            for i in range(hidden_layers):
                if i>=12:
                    padding=1
                self.layers.append(nn.Sequential(ChannelDeconv(min(num_groups,hidden_channels)),nn.Conv2d(hidden_channels,hidden_channels+4,kernel_size,padding=padding),nn.ReLU()))
                hidden_channels=hidden_channels+4
            self.layers.append(nn.Sequential(ChannelDeconv(min(num_groups,hidden_channels)),nn.Conv2d(hidden_channels,num_outputs,kernel_size)))
            """
        elif method==3:
            num_groups=16
            self.layers.append(nn.Sequential(DeConv2d(channels_in,hidden_channels,kernel_size,num_groups=min(num_groups,channels_in)),nn.ReLU()))
            padding = 0
            for i in range(hidden_layers):
                if i>=12:
                    padding=1
                self.layers.append(nn.Sequential(DeConv2d(hidden_channels,hidden_channels+4,kernel_size,padding=padding,num_groups=min(num_groups,hidden_channels)),nn.ReLU()))
                hidden_channels=hidden_channels+4
            self.layers.append(nn.Sequential(DeConv2d(hidden_channels,num_outputs,kernel_size,num_groups=min(num_groups,hidden_channels))))

        elif method==4:
            self.layers.append(nn.Sequential(DeConv2d(channels_in,hidden_channels,kernel_size,mode=2),nn.ReLU()))
            padding = 0
            for i in range(hidden_layers):
                if i>=12:
                    padding=1
                self.layers.append(nn.Sequential(DeConv2d(hidden_channels,hidden_channels+4,kernel_size,padding=padding,mode=2),nn.ReLU()))
                hidden_channels=hidden_channels+4
            self.layers.append(nn.Sequential(DeConv2d(hidden_channels,num_outputs,kernel_size,mode=2)))

        print(hidden_channels,'hidden channels in the final layer.')

    def forward(self, x):
        N, C, H, W = x.shape
        encode=[x]
        for i in range(len(self.layers)):
            x=self.layers[i](x)
            encode.append(x)
        out=F.avg_pool2d(x,x.shape[-2:])
        #print([e.abs().max().item() for e in encode] )
        #print([e.var().item() for e in encode])
        return out.view(out.shape[0],-1)





class MLP(nn.Module):
    def __init__(self,input_nodes, num_outputs=10,hidden_nodes=128,hidden_layers=3,method=1):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        if method == 1:

            self.layers.append(
                nn.Sequential(nn.Linear(input_nodes, hidden_nodes, bias=True)))
            for i in range(hidden_layers-2):
                self.layers.append(nn.Sequential(nn.Sigmoid(),nn.Linear(hidden_nodes, hidden_nodes,bias=True)))
            self.linear = nn.Sequential(nn.Sigmoid(),nn.Linear(hidden_nodes, num_outputs, bias=True))

        if method == 2:
            self.layers.append(
                nn.Sequential(nn.Linear(input_nodes, hidden_nodes, bias=False)))

            for i in range(hidden_layers-2):
                self.layers.append(nn.Sequential(nn.BatchNorm1d(hidden_nodes),nn.Sigmoid(),nn.Linear(hidden_nodes, hidden_nodes,bias=False)))
            self.linear = nn.Sequential(nn.BatchNorm1d(hidden_nodes),nn.Sigmoid(),nn.Linear(hidden_nodes, num_outputs, bias=False))

        elif method == 3:
            num_groups=32
            self.layers.append(
                nn.Sequential(ChannelDeconv(num_groups),nn.Linear(input_nodes, hidden_nodes, bias=False)))

            for i in range(hidden_layers - 2):
                self.layers.append(nn.Sequential(nn.Sigmoid(),ChannelDeconv(num_groups),
                                                 nn.Linear(hidden_nodes, hidden_nodes, bias=False)))
            self.linear = nn.Sequential(nn.Sigmoid(),ChannelDeconv(num_groups),
                                        nn.Linear(hidden_nodes, num_outputs, bias=False))
        elif method == 4:
            num_groups=32
            self.layers.append(
                nn.Sequential(ChannelDeconv(num_groups),nn.Linear(input_nodes, hidden_nodes, bias=False)))

            for i in range(hidden_layers - 2):
                self.layers.append(nn.Sequential(nn.BatchNorm1d(hidden_nodes),nn.Sigmoid(),ChannelDeconv(num_groups),
                                                 nn.Linear(hidden_nodes, hidden_nodes, bias=False)))
            self.linear = nn.Sequential(nn.BatchNorm1d(hidden_nodes),nn.Sigmoid(),ChannelDeconv(num_groups),
                                        nn.Linear(hidden_nodes, num_outputs, bias=False))

        print(hidden_nodes, ' hidden nodes in each layer.')
        print(len(self.layers)+1, ' middle layers.')


        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encode=[x.view(x.shape[0],-1)]
        encode.append(self.layers[0](encode[-1]))
        for i, layer in enumerate(self.layers[1:]):
            encode.append(layer(encode[-1]))
        encode.append(self.linear(encode[-1]))
        #print([e.abs().max().item() for e in encode])
        #print([e.var().item() for e in encode])

        return encode[-1]



