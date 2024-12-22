import torch
import os
import glob
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from data.preprocessing import Crop2Patch
from data.data_augmentation import augmentation
import numpy as np

class PatchLoading(Dataset):
    def __init__(self, opts):
        self.opts = opts
        transform = [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5)
        ]
        self.transforom = transforms.Compose(transform)
        self.files_low = sorted(glob.glob(opts.patch_low+"/*.*"))
        self.files_high = sorted(glob.glob(opts.patch_high+"/*.*"))
    
    def __getitem__(self, index):
        #print(index)
        aug_mode = np.random.randint(0, 8)
        low_path = self.files_low[index % len(self.files_low)]
        high_path = self.files_high[index % len(self.files_high)]
        assert os.path.basename(low_path) == os.path.basename(high_path)
        patch_low = self.transforom(augmentation(Image.open(low_path), aug_mode))
        patch_high = self.transforom(augmentation(Image.open(high_path), aug_mode))
        return {
            'low_light_img': patch_low,
            'high_light_img': patch_high
        }

    def __len__(self):
        return min(len(self.files_low), len(self.files_high))

class EvalLoading(Dataset):
    def __init__(self, opts):
        self.opts = opts
        transform = [
            transforms.ToTensor(),
        ]
        self.transforom = transforms.Compose(transform)
        self.files_low = sorted(glob.glob(opts.eval_low+"/*.*"))
        self.files_high = sorted(glob.glob(opts.eval_high+"/*.*"))
        #self.files_r_low = sorted(glob.glob(opts.eval_low_r+"/*.*"))
        #self.files_r_high = sorted(glob.glob(opts.eval_high_r+"/*.*"))
        #self.files_l_high = sorted(glob.glob(opts.eval_high_l+"/*.*"))
        #self.files_l_low = sorted(glob.glob(opts.eval_ill+"/*.*"))
    
    def __getitem__(self, index):
        patch_low = self.transforom(Image.open(self.files_low[index % len(self.files_low)]))
        patch_high = self.transforom(Image.open(self.files_high[index % len(self.files_low)]))
        #patch_r_low = self.transforom(Image.open(self.files_r_low[index % len(self.files_low)]))
        #patch_r_high = self.transforom(Image.open(self.files_r_high[index % len(self.files_low)]))
        #patch_l_high = self.transforom(Image.open(self.files_l_high[index % len(self.files_l_high)]))
        #patch_l_low = self.transforom(Image.open(self.files_l_low[index % len(self.files_low)]))
        #patch_low = patch_low / 255
        #patch_high = patch_high / 255
        return {
            'low_light_img': patch_low,
            'high_light_img': patch_high,
            #'low_light_r': patch_r_low, 
            #'high_light_r': patch_r_high, 
            #'high_light_l': patch_l_high,
            #'low_light_l': patch_l_low
        }
    def __len__(self):
        return min(len(self.files_low), len(self.files_high))


