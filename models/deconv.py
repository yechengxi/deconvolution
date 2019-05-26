import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import conv
from torch.nn.modules.utils import _pair

#import cv2

#This is a reference implementation using im2col, and is not used anywhere else
class Conv2d(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):


        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), 1, False)

        self.kernel_size=kernel_size
        self.dilation=dilation
        self.padding=padding
        self.stride=stride


    def forward(self, x):
        N,C,H,W=x.shape
        out_h=(H+2*self.padding[0]-self.kernel_size[0]+1)//self.stride[0]
        out_w=(W+2*self.padding[0]-self.kernel_size[0]+1)//self.stride[1]
        w=self.weight
        #im2col
        inp_unf = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.stride)
        #matrix multiplication, reshape
        out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2).view(N,-1,out_h,out_w)

        return out_unf


#iteratively solve for inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA=A.norm()
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt


#deconvolve channels
class ChannelDeconv(nn.Module):
    def __init__(self,  num_groups, eps=1e-2,n_iter=5,momentum=0.1,debug=False):
        super(ChannelDeconv, self).__init__()

        self.eps = eps
        self.n_iter=n_iter
        self.momentum=momentum
        self.num_groups = num_groups
        self.debug=debug

        self.register_buffer('running_mean1', torch.zeros(num_groups, 1))
        #self.register_buffer('running_cov', torch.eye(num_groups))
        self.register_buffer('running_deconv', torch.eye(num_groups))
        self.register_buffer('running_mean2', torch.zeros(1, 1))
        self.register_buffer('running_var', torch.ones(1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_shape = x.shape
        if len(x.shape)==2:
            x=x.view(x.shape[0],x.shape[1],1,1)
        if len(x.shape)==3:
            print('Error! Unsupprted tensor shape.')

        N, C, H, W = x.size()
        G = self.num_groups

        #take the first c channels out for deconv
        c=int(C/G)*G
        if c==0:
            print('Error! num_groups should be set smaller.')


        #step 1. remove mean
        if c!=C:
            x1=x[:,:c].permute(1,0,2,3).contiguous().view(G,-1)
        else:
            x1=x.permute(1,0,2,3).contiguous().view(G,-1)

        mean1 = x1.mean(-1, keepdim=True)

        if self.num_batches_tracked==0:
            self.running_mean1.copy_(mean1.detach())
        if self.training:
            self.running_mean1.mul_(1-self.momentum)
            self.running_mean1.add_(mean1.detach()*self.momentum)
        else:
            mean1 = self.running_mean1

        x1=x1-mean1

        #step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if self.training:
            cov = x1 @ x1.t() / x1.shape[1] + self.eps * torch.eye(G, dtype=x.dtype, device=x.device)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked==0:
            #self.running_cov.copy_(cov.detach())
            self.running_deconv.copy_(deconv.detach())

        if self.training:
            #self.running_cov.mul_(1-self.momentum)
            #self.running_cov.add_(cov.detach()*self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            # cov = self.running_cov
            deconv = self.running_deconv

        x1 =deconv@x1

        #reshape to N,c,J,W
        x1 = x1.view(c, N, H, W).contiguous().permute(1,0,2,3)

        # normalize the remaining channels
        if c!=C:
            x_tmp=x[:, c:].view(N,-1)
            mean2=x_tmp.mean()
            var=x_tmp.var()

            if self.num_batches_tracked == 0:
                self.running_mean2.copy_(mean2.detach())
                self.running_var.copy_(var.detach())

            if self.training:
                self.running_mean2.mul_(1 - self.momentum)
                self.running_mean2.add_(mean2.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum)
                self.running_var.add_(var.detach() * self.momentum)
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = (x[:, c:] - mean2) / (var + self.eps).sqrt()
            x1 = torch.cat([x1, x_tmp], dim=1)


        if self.training:
            self.num_batches_tracked.add_(1)

        if len(x_shape)==2:
            x1=x1.view(x_shape)
        return x1




class DeConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True, eps=1e-2, n_iter=5, momentum=0.1, mode=4, num_groups=16,debug=False):
        # mode 1: remove channel correlation then pixel correlation
        # mode 2: only remove pixel correlation
        # mode 3: only channel correlation
        # mode 4: remove channel correlation and pixel correlation together
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.padding=padding
        self.stride=stride
        super(DeConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), 1, bias, padding_mode='zeros')
        #add padding_mode='zeros' for pytorch 1.1

        self.momentum = momentum
        self.mode=mode
        self.n_iter = n_iter
        self.eps = eps

        num_features = self.weight.shape[2] * self.weight.shape[3]#k*k
        if self.mode!=2:
            if num_groups>self.weight.shape[1]:
                num_groups=self.weight.shape[1]
            self.num_groups=num_groups
            if self.mode!=4:
                self.channel_deconv=ChannelDeconv(num_groups,eps=eps,n_iter=n_iter,momentum=momentum,debug=False)
            else:
                num_features*=num_groups

        self.num_features = num_features

        if self.mode!=3:
            self.register_buffer('running_mean', torch.zeros(num_features,1))
            #self.register_buffer('running_cov', torch.eye(num_features))
            self.register_buffer('running_deconv', torch.eye(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x,visualize=False):

        N,C,H,W=x.shape
        out_h=(H+2*self.padding[0]-self.kernel_size[0]+1)//self.stride[0]
        out_w=(W+2*self.padding[0]-self.kernel_size[0]+1)//self.stride[1]


        if self.mode == 1:
            x = self.channel_deconv(x)

        if  self.mode==3:
            x=self.channel_deconv(x)
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, 1)


        if self.mode!=3:
            #1. im2col, reshape

            # N * cols * pixels
            inp_unf = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.stride)

            #(k*k, C*N*H*W) for pixel deconv
            #(k*k*G, C//G*N*H*W) for grouped pixel deconv
            X=inp_unf.permute(1,0,2).contiguous().view(self.num_features,-1)

            #2.subtract mean
            X_mean = X.mean(-1, keepdim=True)

            #track stats for evaluation
            if self.num_batches_tracked==0:
                self.running_mean.copy_(X_mean.detach())
            if self.training:
                self.running_mean.mul_(1-self.momentum)
                self.running_mean.add_(X_mean.detach()*self.momentum)
            else:
                X_mean = self.running_mean

            X = X - X_mean

            #3. calculate COV, COV^(-0.5), then deconv
            if self.training:
                Cov = X / X.shape[1] @ X.t() + self.eps * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)

            #track stats for evaluation
            if self.num_batches_tracked==0:
                #self.running_cov.copy_(Cov.detach())
                self.running_deconv.copy_(deconv.detach())
            if self.training:
                #self.running_cov.mul_(1-self.momentum)
                #self.running_cov.add_(Cov.detach()*self.momentum)
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)
            else:
                #Cov = self.running_cov
                deconv = self.running_deconv

            #deconv
            X_deconv =deconv@X

            #reshape
            X_deconv=X_deconv.view(-1,N,out_h*out_w).contiguous().permute(1,2,0)

            #4. convolve

            if visualize:
                w = torch.zeros(self.weight.shape[1], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3],
                                dtype=x.dtype, device=x.device)
                c=self.weight.shape[1]
                w[torch.arange(c).long(), torch.arange(c).long(), self.weight.shape[2] // 2, self.weight.shape[3] // 2] = 1.
                out_unf = X_deconv.matmul(w.view(w.size(0), -1).t()).transpose(1, 2).view(N, -1, out_h, out_w)
                return out_unf

            w = self.weight
            out_unf = X_deconv.matmul(w.view(w.size(0), -1).t()).transpose(1, 2).view(N,-1,out_h,out_w)
            if self.bias is not None:
                out_unf=out_unf+self.bias.view(1,-1,1,1)

            if self.training:
                self.num_batches_tracked.add_(1)

            return out_unf#.contiguous()

            """
            #4. convolve(slower )
            X_deconv = torch.nn.functional.fold(X_deconv.transpose(1, 2), (H, W), kernel_size=self.kernel_size,padding=self.padding)/self.kernel_size[0]/self.kernel_size[1]
            return F.conv2d(X_deconv, w, None, self.stride, self.padding, self.dilation, 1)
            """



#this version is faster but slightly weaker, it does not remove the mean.
class FastDeconv(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, eps=1e-2, n_iter=5, momentum=0.1, num_groups=16):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.padding=padding
        self.stride=stride
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        super(FastDeconv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), 1, None, padding_mode='zeros')

        num_features = self.kernel_size[0] * self.kernel_size[1]
        if num_groups>in_channels:
            num_groups=in_channels

        self.num_groups=num_groups
        num_features*=num_groups

        self.num_features = num_features

        self.register_buffer('running_deconv', torch.eye(num_features))

    def forward(self, x,visualize=False):

        N,C,H,W=x.shape

        #1. im2col: N x cols x pixels
        inp_unf = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.stride)

        #(k*k*G, C//G*N*H*W) for grouped pixel deconv
        X = inp_unf.permute(1, 0, 2).contiguous().view(self.num_features, -1)

        #3. calculate COV, COV^(-0.5), then deconv
        if self.training:
            Cov = X / X.shape[1] @ X.t() + self.eps * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)

        #track stats for evaluation
        if self.training:
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            deconv = self.running_deconv

        #deconv
        if visualize:
            w=torch.zeros(self.weight.shape[1],self.weight.shape[1],self.weight.shape[2],self.weight.shape[3],dtype=x.dtype,device=x.device)
            c = self.weight.shape[1]
            w[torch.arange(c).long(), torch.arange(c).long(), self.weight.shape[2] // 2, self.weight.shape[3] // 2] = 1.
            w = w.view(w.shape[0], -1).t().contiguous().view(self.num_features, -1)
            w = deconv @ w
            w = w.view(-1, self.weight.shape[1]).t().view(self.weight.shape[1],self.weight.shape[1],self.weight.shape[2],self.weight.shape[3])
            return F.conv2d(x, w.view(self.weight.shape), None, self.stride, self.padding, self.dilation, 1)
        w=self.weight.view(self.weight.shape[0],-1).t().contiguous().view(self.num_features,-1)
        w=deconv@w
        w=w.view(-1,self.weight.shape[0]).t().view(self.weight.shape)
        return F.conv2d(x, w.view(self.weight.shape), None, self.stride, self.padding, self.dilation, 1)
