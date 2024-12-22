import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1)).view(
        in_feat.size()[0], 1, in_feat.size()[2], in_feat.size()[3]
    )
    return in_feat / (norm_factor.expand_as(in_feat) + eps)
    
def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    N = in0.size()[0]
    X = in0.size()[2]
    Y = in0.size()[3]

    return torch.mean(
        torch.mean(
            torch.sum(in0_norm * in1_norm, dim=1).view(N, 1, X, Y), dim=2
        ).view(N, 1, 1, Y),
        dim=3,
    ).view(N)

class PNet(nn.Module):
    """Pre-trained network with all channels equally weighted by default"""

    def __init__(self, pnet_type="vgg", pnet_rand=False, use_gpu=True):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu

        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.Tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        self.scale = torch.Tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)

        self.net = vgg16(pretrained=not self.pnet_rand, requires_grad=False)
        
        self.L = self.net.N_slices

        if use_gpu:
            self.net.cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()

    def forward(self, in0, in1, retPerLayer=False):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        if retPerLayer:
            all_scores = []
        for (kk, out0) in enumerate(outs0):
            cur_score = 1.0 - cos_sim(outs0[kk], outs1[kk])
            if kk == 0:
                val = 1.0 * cur_score
            else:
                val = val + cur_score
            if retPerLayer:
                all_scores += [cur_score]

        if retPerLayer:
            return (val, all_scores)
        else:
            return val

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs",
            ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"],
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        for param in self.parameters():
            param.required_grad = False
        

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

