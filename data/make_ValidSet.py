import os
import glob
import PIL
from PIL import Image
from skimage.io import imread
import numpy as np

patch_low = "/data/wengjian/low-light-enhancement/Ours/dataset/patch_low"
patch_high = "/data/wengjian/low-light-enhancement/Ours/dataset/patch_high"
patch_r = "/data/wengjian/low-light-enhancement/Ours/dataset/reflectance"
patch_l = "/data/wengjian/low-light-enhancement/Ours/dataset/illumination_low"

files_low = sorted(glob.glob(patch_low+"/*.*"))
files_high = sorted(glob.glob(patch_high+"/*.*"))
files_r = sorted(glob.glob(patch_r+"/*.*"))
files_l = sorted(glob.glob(patch_l+"/*.*"))

patch_valid_low = "/data/wengjian/low-light-enhancement/Ours/dataset/validSet/patch_low"
patch_valid_high = "/data/wengjian/low-light-enhancement/Ours/dataset/validSet/patch_high"
patch_valid_r = "/data/wengjian/low-light-enhancement/Ours/dataset/validSet/reflectance"
patch_valid_l = "/data/wengjian/low-light-enhancement/Ours/dataset/validSet/illumination_low"

if not os.path.exists(patch_valid_low):
    os.makedirs(patch_valid_low)
if not os.path.exists(patch_valid_high):
    os.makedirs(patch_valid_high)
if not os.path.exists(patch_valid_r):
    os.makedirs(patch_valid_r)
if not os.path.exists(patch_valid_l):
    os.makedirs(patch_valid_l)

j = 1
for i in range(len(files_low)):
    if i % 100 == 0:
        print(i)
        low = Image.open(files_low[i])
        high = Image.open(files_high[i])
        r = Image.open(files_r[i])
        l = Image.open(files_l[i])

        low.save(os.path.join(patch_valid_low, str(j)+".png"))
        high.save(os.path.join(patch_valid_high, str(j)+".png"))
        r.save(os.path.join(patch_valid_r, str(j)+".png"))
        l.save(os.path.join(patch_valid_l, str(j)+".png"))

        j = j + 1
        
