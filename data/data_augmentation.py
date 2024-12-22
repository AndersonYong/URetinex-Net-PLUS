from operator import imod
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as tfs
def rotate(degrees, image):
    return tfs.RandomRotation(degrees=degrees)(image)
def flip_vertical(image):
    return tfs.RandomVerticalFlip(p=1)(image)
def flip_horizon(image):
    return tfs.RandomHorizontalFlip(p=1)(image)
def rotate_and_flip(degree, image):
    rotate_image = rotate(degree, image)
    return flip_vertical(rotate_image)
def augmentation(image, mode):
    #print("augmentation")
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # 水平翻转
        return flip_horizon(image)
    elif mode == 2:
        # 垂直翻转
        return flip_vertical(image)
    elif mode == 3:
        # 顺时针旋转90度
        return rotate(degrees=(90, 90), image=image)
    elif mode == 4:
        # 顺时针旋转180度
        return rotate(degrees=(180, 180), image=image)
    elif mode == 5:
        # 顺时针旋转270度
        return rotate(degrees=(270, 270), image=image)
    elif mode == 6:
        # rotate 90 degree and flip up and down
        return rotate_and_flip(degree=(90, 90), image=image)
    elif mode == 7:
        # rotate 180 degree and flip
        return rotate_and_flip(degree=(270, 270), image=image)
if __name__ == "__main__":
    image = "/data/wengjian/low-light-enhancement/pami/evaluate_data/low-source/MEF/Lamp.png"
    transform = [
            transforms.ToTensor(),
        ]
    image = Image.open(image)
    transforom = transforms.Compose(transform)
    results = []
    for i in range(8):
        image_pro = transforom(augmentation(image, i))
        image_pro = image_pro.unsqueeze(0)
        results.append(image_pro)
    results = torch.cat(results, dim=0)
    torchvision.utils.save_image(results, "/data/wengjian/low-light-enhancement/pami/data/flip_test.png")

