from turtle import forward
import torch.nn as nn
import torch
from torch.nn.modules.linear import Identity
from network.architecture import *
import math
import torch.nn.functional as F
from network.ESA import ESALayer

class ESAFusion3(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.esa_layer = ESALayer(n_feats=32*3)
        self.re_map_layer = get_conv2d_layer(in_c=32*3, out_c=32, k=1, s=1, p=0)
        self.weight_layer = get_conv2d_layer(in_c=32, out_c=3*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :]), dim=1)
        R_softmax = self.softmax(R)
        G_softmax = self.softmax(G)
        B_softmax = self.softmax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        #print((re_w1[:, 0:1, :, :]+re_w2[:, 0:1, :, :]+re_w3[:, 0:1, :, :]).mean())
        #print((re_w1[:, 1:2, :, :]+re_w2[:, 1:2, :, :]+re_w3[:, 1:2, :, :]).mean())
        #print((re_w1[:, 2:3, :, :]+re_w2[:, 2:3, :, :]+re_w3[:, 2:3, :, :]).mean())
        re_weights = torch.cat((re_w1, re_w2, re_w3), dim=1)
        return re_weights
    
    def forward(self, results_list, High_L, return_weights=False):
        enhance_1 = results_list[0]
        enhance_2 = results_list[1]
        enhance_3 = results_list[2]
        feat_1 = self.leaky_relu(self.conv1(enhance_1))
        feat_2 = self.leaky_relu(self.conv2(enhance_2))
        feat_3 = self.leaky_relu(self.conv3(enhance_3))
        feat_cat = torch.cat((feat_1, feat_2, feat_3), dim=1)
        feat_fusion = self.esa_layer(feat_cat, feat_cat)
        feat = self.leaky_relu(self.re_map_layer(feat_fusion))
        weights = self.weight_layer(feat)
        weights = self.make_weights(weights)
        R_enhance =  enhance_1*weights[:, 0:3, :, :] + \
                     enhance_2*weights[:, 3:6, :, :] + \
                     enhance_3*weights[:, 6:9, :, :]
        fusion_img = R_enhance * High_L
        if return_weights:
            return fusion_img, R_enhance, weights
        else:
            return fusion_img, R_enhance


class ESAFusion5(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv4 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv5 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)

        self.esa_layer = ESALayer(n_feats=32*5)
        self.re_map_layer = get_conv2d_layer(in_c=32*5, out_c=32, k=1, s=1, p=0)
        self.weight_layer = get_conv2d_layer(in_c=32, out_c=5*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        w4 = weights[:, 9:12, :, :]
        w5 = weights[:, 12:15, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :], w4[:, 0:1, :, :], w5[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :], w4[:, 1:2, :, :], w5[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :], w4[:, 2:3, :, :], w5[:, 2:3, :, :]), dim=1)
        R_softmax = self.softmax(R)
        G_softmax = self.softmax(G)
        B_softmax = self.softmax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_w4 = torch.cat((R_softmax[:, 3:4, :, :], G_softmax[:, 3:4, :, :], B_softmax[:, 3:4, :, :]), dim=1)
        re_w5 = torch.cat((R_softmax[:, 4:5, :, :], G_softmax[:, 4:5, :, :], B_softmax[:, 4:5, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3, re_w4, re_w5), dim=1)
        return re_weights
    
    def forward(self, results_list, High_L, return_weights=False):
        enhance_1 = results_list[0]
        enhance_2 = results_list[1]
        enhance_3 = results_list[2]
        enhance_4 = results_list[3]
        enhance_5 = results_list[4]
        feat_1 = self.leaky_relu(self.conv1(enhance_1))
        feat_2 = self.leaky_relu(self.conv2(enhance_2))
        feat_3 = self.leaky_relu(self.conv3(enhance_3))
        feat_4 = self.leaky_relu(self.conv4(enhance_4))
        feat_5 = self.leaky_relu(self.conv5(enhance_5))
        feat_cat = torch.cat((feat_1, feat_2, feat_3, feat_4, feat_5), dim=1)
        feat_fusion = self.esa_layer(feat_cat, feat_cat)
        feat = self.leaky_relu(self.re_map_layer(feat_fusion))
        weights = self.weight_layer(feat)
        weights = self.make_weights(weights)
        R_enhance =  enhance_1*weights[:, 0:3, :, :] + \
                     enhance_2*weights[:, 3:6, :, :] + \
                     enhance_3*weights[:, 6:9, :, :] + \
                     enhance_4*weights[:, 9:12, :, :] + \
                     enhance_5*weights[:, 12:15, :, :]
        fusion_img = R_enhance * High_L
        if return_weights:
            return fusion_img, R_enhance, weights
        else:
            return fusion_img, R_enhance

class ESAFusion7(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv4 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv5 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv6 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv7 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)

        self.esa_layer = ESALayer(n_feats=32*7)
        self.re_map_layer = get_conv2d_layer(in_c=32*7, out_c=32, k=1, s=1, p=0)
        self.weight_layer = get_conv2d_layer(in_c=32, out_c=7*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
    
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        w4 = weights[:, 9:12, :, :]
        w5 = weights[:, 12:15, :, :]
        w6 = weights[:, 15:18, :, :]
        w7 = weights[:, 18:21, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :], w4[:, 0:1, :, :], w5[:, 0:1, :, :], w6[:, 0:1, :, :], w7[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :], w4[:, 1:2, :, :], w5[:, 1:2, :, :], w6[:, 1:2, :, :], w7[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :], w4[:, 2:3, :, :], w5[:, 2:3, :, :], w6[:, 2:3, :, :], w7[:, 2:3, :, :]), dim=1)
        R_softmax = self.softmax(R)
        G_softmax = self.softmax(G)
        B_softmax = self.softmax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_w4 = torch.cat((R_softmax[:, 3:4, :, :], G_softmax[:, 3:4, :, :], B_softmax[:, 3:4, :, :]), dim=1)
        re_w5 = torch.cat((R_softmax[:, 4:5, :, :], G_softmax[:, 4:5, :, :], B_softmax[:, 4:5, :, :]), dim=1)
        re_w6 = torch.cat((R_softmax[:, 5:6, :, :], G_softmax[:, 5:6, :, :], B_softmax[:, 5:6, :, :]), dim=1)
        re_w7 = torch.cat((R_softmax[:, 6:7, :, :], G_softmax[:, 6:7, :, :], B_softmax[:, 6:7, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3, re_w4, re_w5, re_w6, re_w7), dim=1)
        return re_weights
    
    def forward(self, results_list, High_L, return_weights=False):
        enhance_1 = results_list[0]
        enhance_2 = results_list[1]
        enhance_3 = results_list[2]
        enhance_4 = results_list[3]
        enhance_5 = results_list[4]
        enhance_6 = results_list[5]
        enhance_7 = results_list[6]
        feat_1 = self.leaky_relu(self.conv1(enhance_1))
        feat_2 = self.leaky_relu(self.conv2(enhance_2))
        feat_3 = self.leaky_relu(self.conv3(enhance_3))
        feat_4 = self.leaky_relu(self.conv4(enhance_4))
        feat_5 = self.leaky_relu(self.conv5(enhance_5))
        feat_6 = self.leaky_relu(self.conv6(enhance_6))
        feat_7 = self.leaky_relu(self.conv7(enhance_7))
        feat_cat = torch.cat((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7), dim=1)
        feat_fusion = self.esa_layer(feat_cat, feat_cat)
        feat = self.leaky_relu(self.re_map_layer(feat_fusion))
        weights = self.weight_layer(feat)
        weights = self.make_weights(weights)
        R_enhance =  enhance_1*weights[:, 0:3, :, :] + \
                     enhance_2*weights[:, 3:6, :, :] + \
                     enhance_3*weights[:, 6:9, :, :] + \
                     enhance_4*weights[:, 9:12, :, :] + \
                     enhance_5*weights[:, 12:15, :, :] + \
                     enhance_6*weights[:, 15:18, :, :] + \
                     enhance_7*weights[:, 18:21, :, :]
        fusion_img = R_enhance * High_L
        if return_weights:
            return fusion_img, R_enhance, weights
        else:
            return fusion_img, R_enhance

class ESAFusion3_res(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.esa_layer = ESALayer(n_feats=32*3)
        self.re_map_layer = get_conv2d_layer(in_c=32*3, out_c=32, k=1, s=1, p=0)
        self.weight_layer = get_conv2d_layer(in_c=32, out_c=3*3, k=1, s=1, p=0)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
    def make_weights(self, weights):
        w1 = weights[:, 0:3, :, :]
        w2 = weights[:, 3:6, :, :]
        w3 = weights[:, 6:9, :, :]
        R = torch.cat((w1[:, 0:1, :, :], w2[:, 0:1, :, :], w3[:, 0:1, :, :]), dim=1)
        G = torch.cat((w1[:, 1:2, :, :], w2[:, 1:2, :, :], w3[:, 1:2, :, :]), dim=1)
        B = torch.cat((w1[:, 2:3, :, :], w2[:, 2:3, :, :], w3[:, 2:3, :, :]), dim=1)
        R_softmax = self.softmax(R)
        G_softmax = self.softmax(G)
        B_softmax = self.softmax(B)
        re_w1 = torch.cat((R_softmax[:, 0:1, :, :], G_softmax[:, 0:1, :, :], B_softmax[:, 0:1, :, :]), dim=1)
        re_w2 = torch.cat((R_softmax[:, 1:2, :, :], G_softmax[:, 1:2, :, :], B_softmax[:, 1:2, :, :]), dim=1)
        re_w3 = torch.cat((R_softmax[:, 2:3, :, :], G_softmax[:, 2:3, :, :], B_softmax[:, 2:3, :, :]), dim=1)
        re_weights = torch.cat((re_w1, re_w2, re_w3), dim=1)
        return re_weights
    def forward(self, results_list, High_L):
        enhance_1 = results_list[0]
        enhance_2 = results_list[1]
        enhance_3 = results_list[2]
        feat_1 = self.leaky_relu(self.conv1(enhance_1))
        feat_2 = self.leaky_relu(self.conv2(enhance_2))
        feat_3 = self.leaky_relu(self.conv3(enhance_3))
        feat_cat = torch.cat((feat_1, feat_2, feat_3), dim=1)
        feat_fusion = self.esa_layer(feat_cat, feat_cat) + feat_cat
        feat = self.leaky_relu(self.re_map_layer(feat_fusion))
        weights = self.weight_layer(feat)
        weights = self.make_weights(weights)
        R_enhance =  enhance_1*weights[:, 0:3, :, :] + \
                     enhance_2*weights[:, 3:6, :, :] + \
                     enhance_3*weights[:, 6:9, :, :]
        fusion_img = R_enhance * High_L
        return fusion_img, R_enhance
    
