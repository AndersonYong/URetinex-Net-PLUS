import torch
import torch.nn as nn
from torch.nn import init
from network import Math_Module
from network.Math_Module import P, Q
from network.decom import Decom
import os
from loss.loss_function import perceptual_loss
from data.preprocessing import InitailProcessing
import itertools
from utils import define_compositor, define_modelL, define_modelR, load_param4Decom, load_unfolding
from loss.non_ref_loss import *
from utils import *

from utils import imgs_for_each_t
import torchvision
from loss.metric import PSNR, SSIM
from utils import gamma_correction, get_hist
import time

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

class UnfoldingModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # define model
        self.model_R = define_modelR(self.opts)
        self.model_L = define_modelL(self.opts)
        self.model_Decom_low = Decom()
        self.model_Decom_high = Decom()
        self.P = P()
        self.Q = Q()
        
        self.try_to_load_pretrain()
        training_params = itertools.chain(self.model_L.parameters(), self.model_R.parameters())

        # setup optimizer
        self.optimizer_G = torch.optim.Adam(
            training_params,
            lr = self.opts.lr, 
            betas = (0.9, 0.999),
            weight_decay = 0,
        )
        self.scheduler = self.define_multistep_scheduler()

        # define loss [prevent re-initialize the param of vgg
        self.vgg = perceptual_loss().cuda()
        self.psnr = PSNR().cuda()
        self.ssim = SSIM().cuda()
        print(self.model_R)
        print(self.model_L)

    def define_multistep_scheduler(self):
        """ milestones update step """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer = self.optimizer_G,
            milestones = self.opts.milestones,
            gamma = 0.1,
        )
        return scheduler
    
    def try_to_load_pretrain(self):
        # load for decom network and freeze params
        self.model_Decom_low = load_param4Decom(self.model_Decom_low, self.opts.Decom_model_low_path)
        self.model_Decom_high = load_param4Decom(self.model_Decom_high, self.opts.Decom_model_high_path)
        if self.opts.second_stage == "True":
            if os.path.exists(self.opts.pretrain_unfolding_model_path):
                old_model = torch.load(self.opts.pretrain_unfolding_model_path)
                self.model_R.load_state_dict(old_model['state_dict']['model_R'])
                self.model_L.load_state_dict(old_model['state_dict']['model_L'])
                for paramR in self.model_R.parameters():
                    paramR.required_grad = True
                for paramL in self.model_L.parameters():
                    paramL.required_grad = True
                print("*******=====================> loaded unfolding old one_step_model %s"%self.opts.pretrain_unfolding_model_path)
            else:
                print("pretrained unfolding does not exist, check ---> %s"%self.opts.pretrain_unfolding_model_path)
                exit()
        elif self.opts.second_stage == "False":
            print("unfolding network without pretrain")
        else:
            print("key error")
            exit()

    def forward(self, batch, mode="train"):
        # ===============  feed data ====================
        input_low_img = batch["low_light_img"]
        input_high_img = batch["high_light_img"]
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
            input_high_img = input_high_img.cuda()
        if mode =="eval":
            self.eval()
            [P_results, R_results, L_results, Q_results] = [[], [], [], []]
            #enhance_results = [input_low_img]
            [low_results, high_results, P_high] = [None, None, None]
            with torch.no_grad():       
                for t in range(self.opts.round):
                    if t == 0: # init P, Q
                        [P, Q] = self.model_Decom_low(input_low_img)
                        [P_high, Q_high] = self.model_Decom_high(input_high_img)
                    else: # update P, Q
                        w_p = (self.opts.gamma + self.opts.Roffset * t)
                        w_q = (self.opts.lamda + self.opts.Loffset * t)
                        P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                        Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
                    # upate R, L
                    R = self.model_R(r=P, l=Q)
                    L = self.model_L(l=Q)
                    # save internal results
                    P_results.append(P)           
                    R_results.append(R)           
                    L_results.append(one2three(L))             
                    Q_results.append(one2three(Q))   
            R_results.append(R)
            P_results.append(P_high)
            L_results.append(one2three(Q_high))
            Q_results.append(torch.zeros(one2three(Q_high).shape).to(Q_high))
            low_results = R_results + P_results + L_results + Q_results
            self.train()
            return {"ssim": self.ssim(R, P_high), "psnr": self.psnr(R, P_high)}, \
                torch.cat(low_results, dim=0) #, torch.cat(enhance_results, dim=0)
        else: 
            self.optimizer_G.zero_grad()
            return_imgs = {}
            [P_loss, Q_loss, total_loss] = [0., 0., 0.]
            [w_p, w_q] = [0.5, 0.5]
            losses = {}
            
            for i in range(self.opts.round):
                if i == 0:# init P, Q
                    [P, Q] = self.model_Decom_low(input_low_img)
                    [P_high, Q_high] = self.model_Decom_high(input_high_img)
                else: # update P, Q
                    w_p = (self.opts.gamma + self.opts.Roffset * i)
                    w_q = (self.opts.lamda + self.opts.Loffset * i)
                    P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                    Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
                # update R, L
                R = self.model_R(r=P, l=Q)
                L = self.model_L(l=Q)
                
                P_loss = P_loss + nn.MSELoss()(P, R) * w_p 
                Q_loss = Q_loss + nn.MSELoss()(Q, L) * w_q

            losses["ssim"] = self.ssim(im1=R, im2=P_high)
            losses["psnr"] = self.psnr(im1=R, im2=P_high)
            #print(I_enhance)
            # ======================= loss ========================
            losses["P_loss"] = P_loss
            losses["Q_loss"] = Q_loss
            total_loss += self.opts.l_Pconstraint * P_loss
            total_loss += self.opts.l_Qconstraint * Q_loss

            if "RL2" in self.opts.loss_options:
                losses["R_L2_loss"] = nn.MSELoss()(R, P_high)
                total_loss += self.opts.l_R_l2 * losses["R_L2_loss"]
            
            if "Ltv" in self.opts.loss_options:
                L_tv_loss = TVLoss()(L)
                losses["L_tv_loss"] = L_tv_loss
                total_loss += self.opts.l_Ltv * losses["L_tv_loss"]

            if "RSSIM" in self.opts.loss_options:
                losses["R_ssim_loss"] = 1 - self.ssim(R, P_high)
                total_loss += self.opts.l_R_ssim * losses["R_ssim_loss"]
            
            if "RVGG" in self.opts.loss_options:
                losses["R_vgg_loss"] = self.vgg(R, P_high)
                total_loss += self.opts.l_R_vgg * losses["R_vgg_loss"]
                
            losses["total_loss"] = total_loss

            # ========================== optimize ==========================
            total_loss.mean().backward()
            # TODO: clip_grad_norm
            self.optimizer_G.step()
            self.scheduler.step()
            # ==============================================================
            current_lr = self.optimizer_G.param_groups[0]['lr']
            # save other information
            return_imgs["input_low_img"] = input_low_img
            return_imgs["gtR"] = P_high
            return_imgs["pred_R"] = R
            return_imgs["input_high_img"] = input_high_img
            return return_imgs, losses, current_lr

class DecomModel(nn.Module):
    def __init__(self, opts):
        super().__init__()  
        self.opts = opts
        self.decomModel = Decom()
        self.optimizer_D = torch.optim.Adam(
            list(self.decomModel.parameters()),
            lr = 0.0001,
            betas=(0.99, 0.9)
        )
        print(self.decomModel)

    def forward(self, batch, mode="train"):
        # ===============  feed data ====================
        input_img = None
        L_init = None
        assert (self.opts.img_light == "low" or self.opts.img_light == "high")
        if self.opts.img_light == "low":
            input_img = batch["low_light_img"]
            L_init, _ = torch.max(input_img, dim=1)
            L_init = L_init.unsqueeze(1)
        elif self.opts.img_light == "high":
            input_img = batch["high_light_img"]
            L_init, _ = torch.max(input_img, dim=1)
            L_init = L_init.unsqueeze(1)
            L_init = gamma_correction(L_init, 2.2)
        else :
            print("invalid image for decom training")
            exit()
            
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            L_init = L_init.cuda()

        if mode =="eval":
            self.eval()
            R, L = self.decomModel(input_img)
            results = [input_img, R, one2three(L)]
            self.train()
            return torch.cat(results, dim=0)
        else:
            self.optimizer_D.zero_grad()
            # ================== compute graph =================
            R, L = self.decomModel(input_img)
            
            # loss
            losses = {}
            losses["rec_loss"] = nn.L1Loss()(input_img, R * L)
            losses["L_supervised"] = nn.MSELoss()(L, L_init)
            total_loss = losses["rec_loss"] + 0.1 * losses["L_supervised"]
            if self.opts.img_light == "high":
                losses["L_aware"] = L_structure_aware(self.opts)(illumination=L, img=input_img)
                total_loss += 0.1 * losses["L_aware"]
            losses["total_loss"] = total_loss
            total_loss.mean().backward()
            self.optimizer_D.step()
        return losses

class AdjustModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.adjust_model = define_modelA(self.opts)
        self.fusion_model = define_compositor(self.opts)
        # loading decomposition model
        self.model_Decom_low, self.model_Decom_high = load_decom(self.opts)
        
        # loading R; old_model_opts; and L model
        self.unfolding_model_opts, self.model_R, self.model_L = load_unfolding(self.opts)
        self.P = P()
        self.Q = Q()

        training_params = itertools.chain(self.adjust_model.parameters(), self.fusion_model.parameters())
        # setup optimizer
        self.optimizer_A = torch.optim.Adam(
            training_params,
            lr = 0.0001,
            betas=(0.99, 0.9),
            weight_decay = 0
        )
        self.scheduler = self.define_multistep_scheduler()
        if "spatial" in self.opts.adjust_L_loss:
            self.spatial_operator = SpatialConsistencyLoss()
    
    def define_multistep_scheduler(self):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer = self.optimizer_A,
            milestones = self.opts.milestones,
            gamma = 0.5,
        )
        return scheduler

    def get_ratio(self, high_l, low_l):
        bs, c, w, h = low_l.shape
        ratio_maps = []
        for i in range(bs):
            ratio_mean = (high_l[i, :, :, :] / (low_l[i, :, :, :]+0.0001)).mean()
            #assert ratio_mean > 0
            ratio_mean = max(ratio_mean, self.opts.min_ratio)
            assert ratio_mean >= self.opts.min_ratio
            ratio_maps.append(torch.ones((1, c, w, h)).cuda() * ratio_mean)
        return torch.cat(ratio_maps, dim=0)

    def unfolding_inference(self, input_low_img):
        R_results = []
        for i in range(self.unfolding_model_opts.round):
            if i == 0:
                [P, Q] = self.model_Decom_low(input_low_img)
            else:
                w_p = (self.unfolding_model_opts.gamma + self.unfolding_model_opts.Roffset * i)
                w_q = (self.unfolding_model_opts.lamda + self.unfolding_model_opts.Loffset * i)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
            if (i+1) in self.opts.fusion_layers:
                R_results.append(R)
        assert len(R_results) == len(self.opts.fusion_layers)
        return R_results, L
    
    def make_high_L(self, input_high_img):
        if self.model_Decom_high is not None:
            [_, Q_high] = self.model_Decom_high(input_high_img)
        else:
            Q_high, _ =  torch.max(input_high_img, dim=1)
            Q_high = Q_high.unsqueeze(1)
        return Q_high

    def forward(self, batch, mode="train"):
        input_low_img = batch["low_light_img"]
        input_high_img = batch["high_light_img"]
        
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
            input_high_img = input_high_img.cuda()
        if mode == "eval":
            self.eval()
            enhance_results = [input_low_img]
            with torch.no_grad():  
                # unfolding module
                R_results, L = self.unfolding_inference(input_low_img)
                Q_high = self.make_high_L(input_high_img)
                # adjustment module
                ratio = self.get_ratio(high_l=Q_high, low_l=L).to(L)
                High_L = self.adjust_model(l=L, alpha=ratio)
                if self.fusion_model is not None:
                    I_enhance, R_enhance = self.fusion_model(R_results, High_L)
                else:
                    assert len(R_results) == 1
                    I_enhance = R_results[-1] * High_L

                enhance_results.append(R_results[-1])
                enhance_results.append(one2three(L))
                enhance_results.append(one2three(High_L))
                enhance_results.append(I_enhance)
                enhance_results.append(input_high_img)
            self.train()
            return {"ssim": SSIM()(I_enhance, input_high_img), "psnr": PSNR()(I_enhance, input_high_img)}, \
                    torch.cat(enhance_results, dim=0)
        else:
            assert input_low_img.size(2) == self.opts.size and input_low_img.size(3) == self.opts.size
            self.optimizer_A.zero_grad()
            # unfolding module
            with torch.no_grad():
                R_results, L = self.unfolding_inference(input_low_img)
                Q_high = self.make_high_L(input_high_img)
            # adjustment module
            ratio = self.get_ratio(high_l=Q_high, low_l=L).to(L)
            High_L = self.adjust_model(l=L, alpha=ratio)
            if self.fusion_model is not None:
                I_enhance, R_enhance = self.fusion_model(R_results, High_L)
            else:
                assert len(R_results) == 1
                I_enhance = R_results[-1] * High_L
       
            # loss function
            losses = {}
            total_loss = 0
            if "grad" in self.opts.adjust_L_loss:
                [grad_y, grad_x] = TV_grad()(L)
                [grad_y_high, grad_x_high] = TV_grad()(High_L)
                losses["grad"] = nn.L1Loss().cuda()(grad_x, grad_x_high) + nn.L1Loss().cuda()(grad_y, grad_y_high)
                total_loss += self.opts.l_grad * losses["grad"]
            if "rec" in self.opts.adjust_L_loss:
                losses["high_rec"] = nn.MSELoss()(I_enhance, input_high_img)
                total_loss += losses["high_rec"]
            if "spatial" in self.opts.adjust_L_loss:
                losses["spatial"] = self.spatial_operator(i=input_high_img, i_enhance=I_enhance)
                total_loss += self.opts.l_spa * losses["spatial"]
            losses["total_loss"] = total_loss
            lr = self.optimizer_A.param_groups[0]['lr']
            total_loss.mean().backward()
            self.optimizer_A.step()
            self.scheduler.step()
            return losses, lr
