import torch
import numpy as np
import torch.nn as nn
from network.architecture import get_batchnorm_layer, get_conv2d_layer
import torch.nn.functional as F

# using ResNet Backbone
class Adjust_Res(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# using Simple Convolution -- KinD
class Adjust_naive(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=2, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=1, k=5, s=1, p=2)
        #norm = get_batchnorm_layer(opt)
        #self.batch_norm1 = norm(32)
        #self.batch_norm2 = norm(32)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
    def forward(self, l, alpha):
        input = torch.cat([l, alpha], dim=1)
        x = self.conv1(input)              # -1e-5
        #print(x)
        x = self.conv2(self.leaky_relu(x))   # -1e-6
        #print(x)
        x = self.conv3(self.leaky_relu(x))   # -1e-8
        x = self.conv4(self.leaky_relu(x))   # -0.0002基本不更新
        #print(x)
        x = self.relu(x) 
        #x=self.sigmoid(x)
        return x

class naive_sigmoid(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=2, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=1, k=5, s=1, p=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, l, alpha):
        input = torch.cat([l, alpha], dim=1)
        x = self.conv1(input)              # -1e-5
        x = self.conv2(self.leaky_relu(x))   # -1e-6
        x = self.conv3(self.leaky_relu(x))   # -1e-8
        x = self.conv4(self.leaky_relu(x))   # -0.0002基本不更新
        x=self.sigmoid(x)
        return x


class Adjust_naive_res(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=2, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=1, k=5, s=1, p=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, l, alpha):
        input = torch.cat([l, alpha], dim=1)
        x = self.conv1(input)              # -1e-5
        x = self.conv2(self.leaky_relu(x))   # -1e-6
        x = self.conv3(self.leaky_relu(x))   # -1e-8
        x = self.conv4(self.leaky_relu(x))
        return l + self.leaky_relu(x)





class CurveEstimation(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(2,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,1,3,1,1,bias=True) 

        
    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        A = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

        L_adjust = A
        return L_adjust

class CurveEstimation1x1(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(2,number_f,1,1,0,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,1,1,0,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,1,1,0,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,1,1,0,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,1,3,1,1,bias=True) 

        
    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        A = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

        L_adjust = A
        return L_adjust

class CurveEstimation1x1Pool(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(2,number_f,1,1,0,bias=True) 
        self.maxpool1 = nn.MaxPool2d(2)
        self.e_conv2 = nn.Conv2d(number_f,number_f,1,1,0,bias=True) 
        self.maxpool2 = nn.MaxPool2d(2)
        self.e_conv3 = nn.Conv2d(number_f,number_f,1,1,0,bias=True) 
        self.maxpool3 = nn.MaxPool2d(2)
        self.e_conv4 = nn.Conv2d(number_f,number_f,1,1,0,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.up2 = self.deconv(in_c=number_f, out_c=number_f)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.up3 = self.deconv(in_c=number_f, out_c=number_f)
        self.e_conv7 = nn.Conv2d(number_f*2,1,3,1,1,bias=True) 

    def deconv(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_c, out_c,
                                    kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.e_conv1(x))
        p1 = self.maxpool1(x1)
        x2 = self.relu(self.e_conv2(p1))
        p2 = self.maxpool2(x2)
        x3 = self.relu(self.e_conv3(p2))
        xp = self.maxpool3(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x5 = self.up2(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x6 = self.up3(x6)
        A = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

        L_adjust = A
        return L_adjust

class CurveEstimation5x5Pool(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(2,number_f,5,1,2,bias=True) 
        self.maxpool1 = nn.MaxPool2d(2)
        self.e_conv2 = nn.Conv2d(number_f,number_f,5,1,2,bias=True) 
        self.maxpool2 = nn.MaxPool2d(2)
        self.e_conv3 = nn.Conv2d(number_f,number_f,5,1,2,bias=True) 
        self.maxpool3 = nn.MaxPool2d(2)
        self.e_conv4 = nn.Conv2d(number_f,number_f,5,1,2,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,5,1,2,bias=True) 
        self.up2 = self.deconv(in_c=number_f, out_c=number_f)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,5,1,2,bias=True) 
        self.up3 = self.deconv(in_c=number_f, out_c=number_f)
        self.e_conv7 = nn.Conv2d(number_f*2,1,5,1,2,bias=True) 

    def deconv(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_c, out_c,
                                    kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.e_conv1(x))
        p1 = self.maxpool1(x1)
        x2 = self.relu(self.e_conv2(p1))
        p2 = self.maxpool2(x2)
        x3 = self.relu(self.e_conv3(p2))
        xp = self.maxpool3(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x5 = self.up2(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x6 = self.up3(x6)
        A = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

        L_adjust = A
        return L_adjust

class CurveEstimation5x5PoolRes(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(2,number_f,5,1,2,bias=True) 
        self.maxpool1 = nn.MaxPool2d(2)
        self.e_conv2 = nn.Conv2d(number_f,number_f,5,1,2,bias=True) 
        self.maxpool2 = nn.MaxPool2d(2)
        self.e_conv3 = nn.Conv2d(number_f,number_f,5,1,2,bias=True) 
        self.maxpool3 = nn.MaxPool2d(2)
        self.e_conv4 = nn.Conv2d(number_f,number_f,5,1,2,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,5,1,2,bias=True) 
        self.up2 = self.deconv(in_c=number_f, out_c=number_f)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,5,1,2,bias=True) 
        self.up3 = self.deconv(in_c=number_f, out_c=number_f)
        self.e_conv7 = nn.Conv2d(number_f*2,1,5,1,2,bias=True) 

    def deconv(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_c, out_c,
                                    kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.e_conv1(x))
        p1 = self.maxpool1(x1)
        x2 = self.relu(self.e_conv2(p1))
        p2 = self.maxpool2(x2)
        x3 = self.relu(self.e_conv3(p2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x5 = self.up2(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x6 = self.up3(x6)
        A = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

        L_adjust = l + A
        return L_adjust


    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        A = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

        L_adjust = A

class Naive5x5Res(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=2, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv5 = get_conv2d_layer(in_c=64, out_c=32, k=5, s=1, p=2)
        self.conv6 = get_conv2d_layer(in_c=64, out_c=32, k=5, s=1, p=2)
        self.conv7 = get_conv2d_layer(in_c=64, out_c=1, k=5, s=1, p=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.conv6(torch.cat([x2,x5],1)))
        A = self.relu(self.conv7(torch.cat([x1,x6],1)))
        return A + l
class Naive7x7(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=2, out_c=32, k=7, s=1, p=3)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=7, s=1, p=3)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=7, s=1, p=3)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=32, k=7, s=1, p=3)
        self.conv5 = get_conv2d_layer(in_c=64, out_c=32, k=7, s=1, p=3)
        self.conv6 = get_conv2d_layer(in_c=64, out_c=32, k=7, s=1, p=3)
        self.conv7 = get_conv2d_layer(in_c=64, out_c=1, k=7, s=1, p=3)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, l, alpha):
        x = torch.cat([l, alpha], dim=1)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.conv6(torch.cat([x2,x5],1)))
        A = self.relu(self.conv7(torch.cat([x1,x6],1)))
        return A