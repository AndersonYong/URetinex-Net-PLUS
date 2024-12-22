import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Grayscale
from network.pretrained_model import VGG19
from loss.metric import SSIM
from utils import create_kernel_x, create_kernel_y, split_and_unsqueeze
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import exposure
from skimage.exposure import match_histograms


class Reconstruction_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, i, l, r):
        return self.l1(i, l*r)

class perceptual_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = VGG19().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    
    def forward(self, i, i_hat):
        i_fs = self.vgg19(i)
        i_hat_fs = self.vgg19(i_hat)
        loss = 0
        for i in range(0, len(i_fs)):
            loss += self.weights[i] * self.l1(i_fs[i], i_hat_fs[i])
        return loss

class L1loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, i, i_hat):
        return nn.L1Loss()(i, i_hat)

class SSIM_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return SSIM()(x, y, average=True)

class L2loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return nn.MSELoss()(x, y)

class L2loss_sum(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        bs, _, _, _ = x.shape
        return (nn.MSELoss(reduction='sum')(x, y) / bs)
        