from bisect import bisect_right
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import transforms
#from utils import save_TensorImg
import glob
import torchvision
from PIL import Image
import random

class InitailProcessing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, L_initial="naive"):
        v, i = torch.max(x, 1)
        initial_l = v + 1e-4
        if L_initial == "avg":
            print("avg initialization")
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            initial_l = self.pool(initial_l)
        elif L_initial == "max_avg":
            print("max_avg initialization")
            self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            initial_l = self.max_pool(initial_l)
        elif L_initial == "naive":
            print("naive initialization")
        initial_l3 = torch.stack([initial_l, initial_l, initial_l], dim=1)
        initial_r = x/initial_l3
        if torch.max(initial_r) > 1:
            print("out of line")
            exit()
        return initial_r, initial_l.unsqueeze(1)
    

class Crop2Patch(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.nums_patch = 1000
        self.e = 0.0582
        self.var = 0.0019
    
    def bright(self, img):
        if img.mean() >= self.e + self.var:
            return True
        else :
            return False
    def extremly_dark(self, img):
        if img.mean() <= self.e - self.var:
            return True
        else:
            return False

    def get_file_len(self):
        if not os.path.exists("/data/wengjian/low-light-enhancement/pami/LOLdataset/train/low_%d"%self.size):
            os.makedirs("/data/wengjian/low-light-enhancement/pami/LOLdataset/train/low_%d"%self.size)
            os.makedirs("/data/wengjian/low-light-enhancement/pami/LOLdataset/train/high_%d"%self.size)
            return 0
        else:
            count = len(glob.glob("/data/wengjian/low-light-enhancement/pami/LOLdataset/train/low_%d/*.png"%self.size))
            print(count)
            return count

    def forward(self, img_low, img_high):
        #print(img.shape)
        C, H, W = img_low.shape
        # count是为了读取当前已经存了多少
        print(img_low.mean())
        count = self.get_file_len()
        if self.bright(img_low):
            self.nums_patch = 1200
        if self.extremly_dark(img_low):
            self.nums_patch = 800
        for _ in range(self.nums_patch):
            x = random.randint(0, H-self.size)
            y = random.randint(0, W-self.size)
            low_patch = img_low[:, x:x+self.size, y:y+self.size]
            high_patch = img_high[:, x:x+self.size, y:y+self.size]
            save_low_path = os.path.join("/data/wengjian/low-light-enhancement/pami/LOLdataset/train/low_%d"%self.size, "%d.png"%count)
            save_high_path = os.path.join("/data/wengjian/low-light-enhancement/pami/LOLdataset/train/high_%d"%self.size, "%d.png"%count)
            torchvision.utils.save_image(low_patch, save_low_path)
            torchvision.utils.save_image(high_patch, save_high_path)
            count += 1
        
if __name__ == "__main__":
    low_dir = "/data/wengjian/low-light-enhancement/pami/LOLdataset/train/low"
    high_dir = "/data/wengjian/low-light-enhancement/pami/LOLdataset/train/high"
    files_low = sorted(glob.glob(low_dir+"/*.*"))
    transform = [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5)
    ]
    transforom = transforms.Compose(transform)

    for j, file in enumerate(files_low):
        high_path = os.path.join(high_dir, os.path.basename(file))
        pil_img = Image.open(file)
        high_img = Image.open(high_path)
        img_low = transforom(pil_img)
        img_high = transforom(high_img)
        Crop2Patch(size=96)(img_low, img_high)