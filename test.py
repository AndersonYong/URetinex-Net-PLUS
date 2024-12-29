import argparse
from cmath import exp
from genericpath import exists
from lib2to3.pgen2.pgen import NFAState
from unicodedata import normalize
import torch
import torch.nn as nn
from torch.nn import init
from network import Math_Module
from network.Math_Module import P, Q
from network.decom import Decom
import os
import itertools
from utils import load_decom, load_AdjustFusion, load_unfolding
import torchvision
from loss.metric import PSNR, SSIM
import torchvision.transforms as transforms
from utils import gamma_correction, np_save_TensorImg
from PIL import Image
from data.preprocessing import InitailProcessing
from utils import gamma_correction
import time
import glob
import time
import random
from utils import *
import numpy as np
import matplotlib.pyplot as plts

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

"""
@param: the expected evaluate model
@return: unfolding_path, adjust_fusion_path
"""
def loading_corresponding_model(expected_evaluate):
    unfolding_path = None
    adjust_fusion_path = None
    if expected_evaluate == "URetinex-Net++":
        unfolding_path = "./pretrained_model/unfolding/unfolding_model.pth"
        adjust_fusion_path = "./pretrained_model/fusion_enhance/fusion.pth"
    else:
        print("----------------->invalid algo name")
        exit()
    return unfolding_path, adjust_fusion_path

def getParams(decom_low, r, l, adjust, fusion):
    param = 0
    param += param_self_compute(decom_low)
    param += param_self_compute(r)
    param += param_self_compute(l)
    param += param_self_compute(adjust)
    if fusion is not None:
        param += param_self_compute(fusion)
    return param

class Inference(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # select corresponding pretrain-model according to the expected evalaute
        self.opts.pretrain_unfolding_model_path, self.opts.fusion_model_A_path = loading_corresponding_model(self.opts.alg_name)
        # loading R; old_model_opts; and L model
        self.unfolding_model_opts, self.model_R, self.model_L = load_unfolding(self.opts)
        # loading adjustment model, fusion_model
        self.fusion_opts, self.adjust_model, self.fusion_model = load_AdjustFusion(self.opts)
        # loading decomposition model 
        self.model_Decom_low, self.model_Decom_high = load_decom(self.fusion_opts)
    
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
       
        print("total parameters: ", getParams(self.model_Decom_low, self.model_R, self.model_L, self.adjust_model, self.fusion_model))

    def get_ratio(self, high_l, low_l):
        bs, c, w, h = low_l.shape
        ratio_maps = []
        for i in range(bs):
            ratio_mean = (high_l[i, :, :, :] / (low_l[i, :, :, :]+0.0001)).mean()
            if "min_ratio" in self.fusion_opts:
                ratio_mean = max(ratio_mean, self.fusion_opts.min_ratio)
                assert ratio_mean >= self.fusion_opts.min_ratio
            ratio_maps.append(torch.ones((1, c, w, h)).cuda() * ratio_mean)
        return torch.cat(ratio_maps, dim=0)
    
    def make_high_L(self, input_high_img):
        if self.model_Decom_high is not None:
            [R_high, Q_high] = self.model_Decom_high(input_high_img)
        else:
            Q_high, _ =  torch.max(input_high_img, dim=1)
            Q_high = Q_high.unsqueeze(0)
            R_high = input_high_img / (Q_high+0.0001)
        return R_high, Q_high
    
    def ratio_maker(self, L, input_high_img):
        if input_high_img is None:
            ratio = torch.ones(L.shape).cuda() * self.opts.ratio
        else:
            _, Q_high = self.make_high_L(input_high_img)
            ratio = self.get_ratio(high_l=Q_high, low_l=L)
        return ratio
    
    def unfolding(self, input_low_img, return_component=False):
        P_results, Q_results, R_results, L_results = [[], [], [], []]
        for t in range(self.unfolding_model_opts.round):      
            if t == 0: # init P, Q
                P, Q = self.model_Decom_low(input_low_img)
            else: # update P, Q
                w_p = (self.unfolding_model_opts.gamma + self.unfolding_model_opts.Roffset * t)
                w_q = (self.unfolding_model_opts.lamda + self.unfolding_model_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
            # update R, L
            # update R, L
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
            P_results.append(P)
            Q_results.append(Q)
            R_results.append(R)
            L_results.append(L)
        if not return_component:
            return R_results, L_results
        else:
            return P_results, Q_results, R_results, L_results

    def unfolding_inference(self, input_low_img):
        R_results, L_results = self.unfolding(input_low_img, return_component=False)
        results = []
        for t in range(len(R_results)):
            if (t+1) in self.fusion_opts.fusion_layers:
                #print("fusion layer : %d"%(t+1))
                results.append(R_results[t])
        assert len(results) == len(self.fusion_opts.fusion_layers)
        return results, L_results[-1]

    def adjust_and_fusion(self, inter_R_list, L, ratio):
        High_L = self.adjust_model(l=L, alpha=ratio)
        R_enhance = None
        if self.fusion_model is not None:
            I_enhance, R_enhance = self.fusion_model(inter_R_list, High_L)
        else:
            assert len(inter_R_list) == 1
            I_enhance = inter_R_list[-1] * High_L
        return High_L, I_enhance, R_enhance
    
    def forward(self, input_low_img, input_high_img=None):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
            if input_high_img is not None:
                input_high_img = input_high_img.cuda()
        with torch.no_grad(): 
            start = time.time() 
            R_results, L = self.unfolding_inference(input_low_img=input_low_img)
            ratio = self.ratio_maker(L, input_high_img)
            High_L, I_enhance, R_enhance = self.adjust_and_fusion(inter_R_list=R_results, L=L, ratio=ratio) 
            p_time = (time.time() - start)
        return I_enhance, p_time
    
    def get_corresponding_high(self, ratio, low_path, dataset):
        if ratio is None:
            high_img_path = os.path.join(self.opts.real_high+"/%s_gt"%dataset, os.path.basename(low_path))
            high_img = self.transform(Image.open(high_img_path)).unsqueeze(0).cuda()
        else:
            high_img = None
        return high_img
    
    def ProcessAndTime(self, low_img, high_img):
        time_each_img = []
        for i in range(self.opts.loop_time_for_img):
            enhance, p_time = self.forward(input_low_img=low_img, input_high_img=high_img)
            if i!= 0:
                # ignore the first one.
                time_each_img.append(p_time)
        return enhance, np.mean(time_each_img)

    def run(self):
        datasets = ["LOL"]
        for dataset in datasets:
            data_dir = os.path.join(self.opts.low_dir, dataset)
            files_low = sorted(glob.glob(data_dir+"/*.*"))
            img_save_root = os.path.join(self.opts.save_dir, dataset, self.opts.alg_name)
            if not os.path.exists(img_save_root): 
                os.makedirs(img_save_root)
            time_each_dataset = []
            for (low_img_path) in (files_low):
                print("processing ---> %s"%low_img_path)
                low_img = self.transform(Image.open(low_img_path)).unsqueeze(0)
                # get file name
                file_name = os.path.basename(low_img_path)
                name = file_name.split('.')[0]
                # get corresonding high_img
                high_img = self.get_corresponding_high(self.opts.ratio, low_img_path, dataset)
                # define save_path
                save_path = os.path.join(img_save_root, file_name.replace(name, "%s__%s"%(name,self.opts.alg_name)))
                if self.opts.test == "metric":
                    enhance, time_each_img = self.ProcessAndTime(low_img, high_img)
                    time_each_dataset.append(np.mean(time_each_img))
                    np_save_TensorImg(enhance, save_path) 
            print("=================================  average time for dataset %s: %f============================"%(dataset, np.mean(time_each_dataset)))
    
    def run_one_image(self, low_img_path, output_dir):
        assert self.opts.ratio is not None
        # loading img
        low_img = self.transform(Image.open(low_img_path)).unsqueeze(0)
        file_name = os.path.basename(low_img_path)
        name = file_name.split('.')[0]
        # inference
        print("processing ---> %s"%low_img_path)
        time_each_img = []
        for i in range(self.opts.loop_time_for_img):
            enhance, p_time = self.forward(input_low_img=low_img, input_high_img=None)
            if i!= 0:
                time_each_img.append(p_time)
        print("=================================  average time for %s: %f============================"%(name, np.mean(time_each_img)))
        # save the enhanced image
        save_path = os.path.join(output_dir, "%s__%s.png"%(name, self.opts.alg_name))
        torchvision.utils.save_image(enhance, save_path)
  
    def run_darkface(self, low_dir, output_dir):
        output_dir = os.path.join(output_dir, "Alg_%s_%0.1f"%(self.opts.alg_name, self.opts.ratio))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        low_lists = glob.glob(os.path.join(low_dir, "*.png"))
        assert len(low_lists) == 500
        for low_file in low_lists:
            self.run_one_image(low_file, output_dir)
    
    def run_and_evaluate(self, evaluate=False):
        self.run()
        if evaluate and self.opts.test=="metric":
            from get_metric import evaluate_all
            datasets = ["LOL"]  # add datatset here
            for dataset in datasets:
                evaluate_all(dataset, self.opts.alg_name, self.opts, file_path="./evaluate_metric.txt")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    parser.add_argument('--real_high', type=str, default="./evaluate_data/real_high")
    parser.add_argument('--low_dir', type=str, default="./evaluate_data/low-source")
    parser.add_argument('--gpu_id', type=str, default="4")
    parser.add_argument('--loop_time_for_img', type=int, default=1)
    parser.add_argument('--alg_name', type=str, default="URetinex-Net++")
    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--evaluate', action="store_true", default=True)
    parser.add_argument('--test',type=str, default="metric")
    parser.add_argument('--save_dir', type=str, default="./evaluate_img_results")
    parser.add_argument('--high_dir_root', type=str, default="./evaluate_data/real_high/")
    # model path
    parser.add_argument('--Decom_model_low_path', type=str, default="./pretrained_model/decom/decom_low_light.pth")
    parser.add_argument('--Decom_model_high_path', type=str, default="./pretrained_model/decom/decom_high_light.pth")
    
    parser.add_argument('--pretrain_unfolding_model_path', type=str, default=None)
    parser.add_argument('--fusion_model_A_path', type=str, default=None)

    opts = parser.parse_args()
    for k, v in vars(opts).items():
        print(k, v)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    model = Inference(opts).cuda()
    model.run_and_evaluate(evaluate=opts.evaluate)
    #model.run_one_image("/data/wengjian/low-light-enhancement/pami/evaluate_data/low-source/LIME/9.png", 
    #        output_dir="/data/wengjian/low-light-enhancement/pami/evaluate_data/enhance_pami")