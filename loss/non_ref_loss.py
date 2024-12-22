import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Grayscale
from network.pretrained_model import VGG19
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """ encourages spatial coherence ofthe enhanced image 
        through preserving the difference between input and enhanced ones
    """
    def __init__(self):
        super().__init__()
        kernel_left = torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0,-1,0],[0, 1, 0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0,0,0],[0, 1, 0],[0,-1,0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, i, i_enhance):
        b, c, h, w = i.shape
        i_mean = torch.mean(i, 1, keepdim=True)
        i_enhance_mean = torch.mean(i_enhance, 1, keepdim=True)
        i_pool =  self.pool(i_mean)			
        enhance_pool = self.pool(i_enhance_mean)

        D_i_left = F.conv2d(i_pool, self.weight_left, padding=1)
        D_i_right = F.conv2d(i_pool, self.weight_right, padding=1)
        D_i_up = F.conv2d(i_pool, self.weight_up, padding=1)
        D_i_down = F.conv2d(i_pool, self.weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_i_left - D_enhance_left,2)
        D_right = torch.pow(D_i_right - D_enhance_right,2)
        D_up = torch.pow(D_i_up - D_enhance_up,2)
        D_down = torch.pow(D_i_down - D_enhance_down,2)
        return torch.mean(D_left + D_right + D_up +D_down)


class TVLoss(nn.Module):
    """ illumination to be smoothness"""
    def __init__(self):
        super().__init__()
    def forward(self, illumination):
        B, C, H, W = illumination.shape
        count_h =  (H-1) * W
        count_w = H * (W-1)
        h_tv = torch.pow((illumination[:, :, 1:, :]-illumination[:, :, :H-1, :]), 2).sum()
        w_tv = torch.pow((illumination[:, :, :, 1:]-illumination[:, :, :, :W-1]), 2).sum()
        return (h_tv/count_h + w_tv/count_w) / B

def convert2gray(input_tensor):
    if input_tensor.size(1) == 3:
        r = input_tensor[:, 0:1, :, :]
        g = input_tensor[:, 1:2, :, :]
        b = input_tensor[:, 2:3, :, :]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        return input_tensor

class L_structure_aware(nn.Module):
    """ the illumination total variation weighted by grad of reflectance map
        make it struture aware
        this method compare to LIME(TIP 2017), the weight can be updated simulaneously
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.c = 10
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    def gradient(self, x):
        b, c, h, w = x.shape
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :h-1, :])
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w-1])
        return grad_y, grad_x
    
    def forward(self, illumination, img):
        img_gray = convert2gray(img)
        grad_l_y, grad_l_x = self.gradient(illumination)
        grad_r_y, grad_r_x = self.gradient(img_gray)
        avg_grad_r_y = self.pool(grad_r_y)
        avg_grad_r_x = self.pool(grad_r_x)
        x_tv = grad_l_x * torch.exp(-self.c * avg_grad_r_x)
        y_tv = grad_l_y * torch.exp(-self.c * avg_grad_r_y)
        return torch.mean(x_tv) + torch.mean(y_tv)

        
class TV_grad(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b, c, h, w = x.shape
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :h-1, :])
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w-1])
        return grad_y, grad_x

class Illumination_smooth(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.epsilon = 0.01
    def gradient(self, x):
        b, c, h, w = x.shape
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :h-1, :])
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w-1])
        [grad_y_max, grad_y_min] = [grad_y.max(), grad_y.min()]
        [grad_x_max, grad_x_min] = [grad_x.max(), grad_x.min()]
        grad_y_norm = (grad_y - grad_y_min) / (grad_y_max - grad_y_min + 0.0001)
        grad_x_norm = (grad_x - grad_x_min) / (grad_x_max - grad_x_min + 0.0001)
        return grad_y_norm, grad_x_norm
    def forward(self, l, i):
        img_gray = convert2gray(i)
        l_grad_y, l_grad_x = self.gradient(l)
        i_grad_y, i_grad_x = self.gradient(img_gray)
        eposilon_x = self.epsilon * torch.ones(i_grad_x.shape).to(i_grad_x)
        eposilon_y = self.epsilon * torch.ones(i_grad_y.shape).to(i_grad_y)
        x_loss = torch.abs((l_grad_x) / (torch.max(i_grad_x, eposilon_x)))
        y_loss = torch.abs((l_grad_y) / (torch.max(i_grad_y, eposilon_y)))
        return x_loss.mean() + y_loss.mean()


