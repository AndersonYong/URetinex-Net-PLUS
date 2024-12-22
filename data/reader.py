import torch
import os
import glob
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from data.preprocessing import Crop2Patch

class MyDataset(Dataset):
    def __init__(self, opts):
        self.opts = opts
        root = opts.root
        transform = [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5)
        ]
        self.transforom = transforms.Compose(transform)
        self.files_low = sorted(glob.glob(os.path.join(root, opts.dataset)+"/low/*.*"))
        self.files_high = sorted(glob.glob(os.path.join(root, opts.dataset)+"/high/*.*"))
    
    def __getitem__(self, index):
        
        item_low = self.transforom(Image.open(self.files_low[index % len(self.files_low)]))
        item_high = self.transforom(Image.open(self.files_high[index % len(self.files_low)]))
        return {
            'low_light_image': item_low,
            'high_light_image': item_high
        }

    def __len__(self):
        return min(len(self.files_low), len(self.files_high))