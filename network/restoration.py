from turtle import forward
import torch.nn as nn
import torch
from torch.nn.modules.linear import Identity
from network.architecture import *
import math
import torch.nn.functional as F
from network.architecture import ResBlock, Identity

class HalfDnCNNSE(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        
        if self.opts.concat_L:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = get_conv2d_layer(in_c=1, out_c=32, k=3, s=1, p=1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = self.conv1 = get_conv2d_layer(in_c=3, out_c=64, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
        self.se_layer = SELayer(channel=64)
        self.conv3 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = get_conv2d_layer(in_c=64, out_c=3, k=3, s=1, p=1)

    def forward(self, r, l):
        if self.opts.concat_L:
            r_fs = self.relu1(self.conv1(r))
            l_fs = self.relu2(self.conv2(l))
            inf = torch.cat([r_fs, l_fs], dim=1)
            se_inf = self.se_layer(inf)
        else:
            r_fs = self.relu1(self.conv1(r))
            se_inf = self.se_layer(r_fs)
        x1 = self.relu3(self.conv3(se_inf))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        n = self.conv8(x5)
        r_restore = r + n
        return r_restore


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 全局取平均 == mean(3, True).mean(2, True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SECompositor_3(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.se_layer = SELayer(channel=32*3)
        self.rgb_layer = get_conv2d_layer(in_c=32*3, out_c=3, k=1, s=1, p=0)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inter_R_list):
        feat_1 = self.relu(self.conv1(inter_R_list[0]))
        feat_2 = self.relu(self.conv2(inter_R_list[1]))
        feat_3 = self.relu(self.conv3(inter_R_list[2]))
        feat_fusion = self.se_layer(torch.cat((feat_1, feat_2, feat_3), dim=1))
        fusion_rgb = self.relu(self.rgb_layer(feat_fusion))
        return fusion_rgb

class SoftMaxSECompositor_3(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.se_layer = SELayer(channel=32*3)
        self.weight_layer = get_conv2d_layer(in_c=32*3, out_c=3*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :]), dim=1)
        #SoftMax = nn.Softmax(dim=1)
        R_softmax = self.softmax(R)
        G_softmax = self.softmax(G)
        B_softmax = self.softmax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3), dim=1)
        return re_weights
        
    def forward(self, inter_R_list):
        feat_1 = self.leaky_relu(self.conv1(inter_R_list[0]))
        feat_2 = self.leaky_relu(self.conv2(inter_R_list[1]))
        feat_3 = self.leaky_relu(self.conv3(inter_R_list[2]))
        feat_fusion = self.se_layer(torch.cat((feat_1, feat_2, feat_3), dim=1))

        weights = self.weight_layer(feat_fusion)
        weights = self.make_weights(weights)
        R_fusion = inter_R_list[0]*weights[:, 0:3, :, :] + inter_R_list[1]*weights[:, 3:6, :, :] + inter_R_list[2]*weights[:, 6:9, :, :]
        return R_fusion

class SoftMaxSECompositor_3_Dilation(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=3, dilation=3)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=2, dilation=2)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1, dilation=1)
        self.se_layer = SELayer(channel=32*3)
        self.weight_layer = get_conv2d_layer(in_c=32*3, out_c=3*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :]), dim=1)
        #SoftMax = nn.Softmax(dim=1)
        R_softmax = self.softmax(R)
        G_softmax = self.softmax(G)
        B_softmax = self.softmax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3), dim=1)
        return re_weights
        
    def forward(self, inter_R_list):
        feat_1 = self.leaky_relu(self.conv1(inter_R_list[0]))
        feat_2 = self.leaky_relu(self.conv2(inter_R_list[1]))
        feat_3 = self.leaky_relu(self.conv3(inter_R_list[2]))
        feat_fusion = self.se_layer(torch.cat((feat_1, feat_2, feat_3), dim=1))

        weights = self.weight_layer(feat_fusion)
        weights = self.make_weights(weights)
        R_fusion = inter_R_list[0]*weights[:, 0:3, :, :] + inter_R_list[1]*weights[:, 3:6, :, :] + inter_R_list[2]*weights[:, 6:9, :, :]
        return R_fusion

class SoftMaxSECompositor_2(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.se_layer = SELayer(channel=32*2)
        self.weight_layer = get_conv2d_layer(in_c=32*2, out_c=2*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :]), dim=1)
        SoftMax = nn.Softmax(dim=1)
        R_softmax = SoftMax(R)
        G_softmax = SoftMax(G)
        B_softmax = SoftMax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2), dim=1)
        return re_weights
        
    def forward(self, inter_R_list):
        feat_1 = self.leaky_relu(self.conv1(inter_R_list[0]))
        feat_2 = self.leaky_relu(self.conv2(inter_R_list[1]))
        feat_fusion = self.se_layer(torch.cat((feat_1, feat_2), dim=1))

        weights = self.weight_layer(feat_fusion)
        weights = self.make_weights(weights)
        R_fusion = inter_R_list[0]*weights[:, 0:3, :, :] + inter_R_list[1]*weights[:, 3:6, :, :]
        return R_fusion

class SoftMaxSECompositor_5(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv4 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv5 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.se_layer = SELayer(channel=32*5)
        self.weight_layer = get_conv2d_layer(in_c=32*5, out_c=5*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        w4 = weights[:, 9:12, :, :]
        w5 = weights[:, 12:15, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :], w4[:, 0:1, :, :], w5[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :], w4[:, 1:2, :, :], w5[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :], w4[:, 2:3, :, :], w5[:, 2:3, :, :]), dim=1)
        SoftMax = nn.Softmax(dim=1)
        R_softmax = SoftMax(R)
        G_softmax = SoftMax(G)
        B_softmax = SoftMax(B)
        #print(R_softmax.sum(dim=1).mean())
        
        assert R_softmax.sum(dim=1).mean()==1
        assert G_softmax.sum(dim=1).mean()==1
        assert B_softmax.sum(dim=1).mean()==1
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_w4 = torch.cat((R_softmax[:, 3:4, :, :], G_softmax[:, 3:4, :, :], B_softmax[:, 3:4, :, :]), dim=1)
        re_w5 = torch.cat((R_softmax[:, 4:5, :, :], G_softmax[:, 4:5, :, :], B_softmax[:, 4:5, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3, re_w4, re_w5), dim=1)
        return re_weights
        
    def forward(self, inter_R_list):
        feat_1 = self.leaky_relu(self.conv1(inter_R_list[0]))
        feat_2 = self.leaky_relu(self.conv2(inter_R_list[1]))
        feat_3 = self.leaky_relu(self.conv3(inter_R_list[2]))
        feat_4 = self.leaky_relu(self.conv4(inter_R_list[3]))
        feat_5 = self.leaky_relu(self.conv5(inter_R_list[4]))
        feat_fusion = self.se_layer(torch.cat((feat_1, feat_2, feat_3, feat_4, feat_5), dim=1))
        weights = self.weight_layer(feat_fusion)
        weights = self.make_weights(weights)
        R_fusion = inter_R_list[0]*weights[:, 0:3, :, :] + inter_R_list[1]*weights[:, 3:6, :, :] + inter_R_list[2]*weights[:, 6:9, :, :] +\
                        inter_R_list[3] * weights[:, 9:12, :, :] + inter_R_list[4] * weights[:, 12:15, :, :]
        return R_fusion

class SoftMaxSECompositor_4(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv4 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.se_layer = SELayer(channel=32*4)
        self.weight_layer = get_conv2d_layer(in_c=32*4, out_c=4*3, k=3, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        w4 = weights[:, 9:12, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :], w4[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :], w4[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :], w4[:, 2:3, :, :]), dim=1)
        SoftMax = nn.Softmax(dim=1)
        R_softmax = SoftMax(R)
        G_softmax = SoftMax(G)
        B_softmax = SoftMax(B)
        #print(R_softmax.sum(dim=1).mean())
        
        assert R_softmax.sum(dim=1).mean()==1
        assert G_softmax.sum(dim=1).mean()==1
        assert B_softmax.sum(dim=1).mean()==1
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_w4 = torch.cat((R_softmax[:, 3:4, :, :], G_softmax[:, 3:4, :, :], B_softmax[:, 3:4, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3, re_w4), dim=1)
        return re_weights
        
    def forward(self, inter_R_list):
        feat_1 = self.leaky_relu(self.conv1(inter_R_list[0]))
        feat_2 = self.leaky_relu(self.conv2(inter_R_list[1]))
        feat_3 = self.leaky_relu(self.conv3(inter_R_list[2]))
        feat_4 = self.leaky_relu(self.conv4(inter_R_list[3]))
        feat_fusion = self.se_layer(torch.cat((feat_1, feat_2, feat_3, feat_4), dim=1))
        weights = self.weight_layer(feat_fusion)
        weights = self.make_weights(weights)
        R_fusion = inter_R_list[0]*weights[:, 0:3, :, :] + inter_R_list[1]*weights[:, 3:6, :, :] + inter_R_list[2]*weights[:, 6:9, :, :] +\
                        inter_R_list[3] * weights[:, 9:12, :, :]
        return R_fusion