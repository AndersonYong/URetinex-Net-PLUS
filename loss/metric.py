import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from network.pretrained_model import PNet
import lpips

class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 11
        
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        )
        return window
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
        )
        return gauss / gauss.sum()

    def _ssim(self, im1, im2, window, window_size, channel, average):
        mu1 = F.conv2d(im1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(im2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(im1 * im1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(im2 * im2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(im1 * im2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        up = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        down = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map = up / down

        if average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, im1, im2, average=True):
        channel = im1.size(1)
        window = self.create_window(self.window_size, channel)
        if im1.is_cuda:
            window = window.cuda(im1.get_device())
        window = window.type_as(im1)
        return self._ssim(im1, im2, window, self.window_size, channel, average)

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, im1, im2):
        #bs = im1.size(0)
        #mse_err = (im1 - im2).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)
        #psnr = 10 * (1 / mse_err).log10()
        # assert im1.size(0) == 1 and im2.size(0) == 1  # gsb 2024/4/23 注释掉
        a = torch.log10(1. * 1. / nn.MSELoss()(im1, im2)) * 10
        psnr = torch.clamp(a, 0., 99.99)
        return psnr.mean()

class perceptual_sim(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = PNet()
        self.vgg16.eval().cuda()

    def forward(self, im1, im2):
        return self.vgg16(im1*2-1, im2*2-1).mean()

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, im1, im2):
        return nn.MSELoss()(im1, im2)

class DSSIM(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, im1, im2):
        return ((1 - SSIM()(im1, im2)) / 2)

class L1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, im1, im2):
        return nn.L1Loss()(im1, im2)

class LPIPS_ofical(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', lpips=True)
    def forward(self, im1, im2):
        return self.loss_fn_vgg(im1*2-1, im2*2-1).mean()
        
class Angular_error(nn.Module):
    def __init__(self):
        super().__init__()
    """
        cos<x, y> = <x, y> / |x|*|y|
        F.normalize(tensor, dim) == tensor / |tensor|
    """
    def forward(self, x, y):
        bs, c, h, w = x.shape
        x_vec = x.view(bs, c, h*w).permute(0, 2, 1)  # --> (bs, h*w, 3)
        y_vec = y.view(bs, c, h*w).permute(0, 2, 1)  # --> (bs, h*w, 3)
        x_vec_normalize = F.normalize(x_vec, p=2, dim=2)
        y_vec_normalize = F.normalize(y_vec, p=2, dim=2)
        cos_value = x_vec_normalize * y_vec_normalize
        cos_value = torch.sum(cos_value, dim=2)
        cos_value = torch.clamp(cos_value, min=-1, max=1)
        angle = torch.acos(cos_value)
        return torch.sum(angle)

# TODO NIQE; LOE; LOE_ref; TMQI; LMSE