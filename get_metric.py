from loss.metric import PSNR, SSIM, perceptual_sim, Angular_error, LPIPS_ofical
from loss.DISTS_pytorch.DISTS_pt import DISTS
from utils import save_TensorImg, write_config, write_imgs, write_loss, write_config_to_file, write_metric_to_file
import os
import torch
import glob
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def write_metric_to_file(method, metric_dicts, file_path):
    with open(file_path, 'a') as f:
        f.write(str(method) + "       " + time.asctime(time.localtime(time.time())) + '\n')
        for metric in metric_dicts.keys():
            f.write("%s: "%metric + "     "+str(round(metric_dicts[metric], 4)) + '\n')
        f.write('\n')
        f.close()
def write_dataset(dataset, file_path):
    with open(file_path, 'a') as f:
        f.write("======================= %s ========================\n"%dataset)
        f.close()

def write_psnr(method, psnr_list, file_path):
    with open(file_path, 'a') as f:
        f.write(str(method) +  "       " + time.asctime(time.localtime(time.time())) + '\n')
        f.write(str(psnr_list))
        f.write('\n')
        f.close()

def crop2KinD_plus(high_img):
    bs, c, h, w  = high_img.shape
    h_tmp = h % 4
    w_tmp = w % 4
    high_img_resize = high_img[:, :, 0:h-h_tmp, 0:w-w_tmp]
    return high_img_resize


def evaluate(pred_dir, gt_dir, transform, alg_name, dataset, mean=True):
    [ssim, psnr, lpips_cos, lpips, mae, dists, angular, vif] = [[], [], [], [], [], [], [], []]
    files_fake = sorted(glob.glob(os.path.join(pred_dir,"*.*")))
    files_high = sorted(glob.glob(os.path.join(gt_dir, "*.*")))
    # define eval model
    mae_tool = torch.nn.L1Loss().cuda()
    ssim_tool = SSIM().cuda()
    psnr_tool = PSNR().cuda()
    lpipsCos_tool = perceptual_sim().cuda()
    lpips_tool = LPIPS_ofical().cuda()
    dists_tool = DISTS().cuda()
    # define metric dicts ['metric': value]
    metric_dicts = {}
    for (fake_img_path, high_img_path) in zip(files_fake, files_high):
        file_name = os.path.basename(fake_img_path).split('_')[0]
        high_img_path = os.path.join(os.path.dirname(high_img_path), os.path.basename(high_img_path).replace(os.path.basename(high_img_path).split('.')[0], file_name))
        print(fake_img_path)
        print(high_img_path)
        assert os.path.basename(high_img_path).split('.')[0] == os.path.basename(fake_img_path).split('_')[0]
        # reading data
        fake_img = transform(Image.open(fake_img_path)).unsqueeze(0).cuda()
        high_img = transform(Image.open(high_img_path)).unsqueeze(0).cuda()
        # special handling for KIND++
        if alg_name == "KinD++":
            high_img = crop2KinD_plus(high_img)
        print(high_img.shape)
        print(fake_img.shape)
        assert high_img.shape == fake_img.shape
        assert fake_img.size(0) == 1
        ssim.append(ssim_tool(fake_img, high_img).item())
        psnr.append(psnr_tool(fake_img, high_img).item())
        lpips_cos.append(lpipsCos_tool(fake_img, high_img).item())
        lpips.append(lpips_tool(fake_img, high_img).item())
        mae.append(mae_tool(fake_img, high_img).item())
        dists.append(dists_tool(fake_img, high_img).item())

    if mean:
        metric_dicts["MAE"] = np.mean(mae)
        metric_dicts["SSIM"] = np.mean(ssim)
        metric_dicts["PSNR"] = np.mean(psnr)
        metric_dicts["LPIPS_COS"] = np.mean(lpips_cos)
        metric_dicts["LPIPS"] = np.mean(lpips)
        metric_dicts["DISTS"] = np.mean(dists)
        print(np.mean(psnr))
        return metric_dicts
    else:
        return mae, ssim, psnr, lpips_cos, lpips, dists

def evaluate_all(dataset, alg_name, opts, file_path):
    transform = [transforms.ToTensor()]
    transform = transforms.Compose(transform)
    # get gt
    gt_dir = os.path.join(opts.high_dir_root, dataset+"_gt")
    # get predict imgs
    pred_dir = os.path.join(opts.save_dir, dataset, alg_name)
    if not os.path.exists(pred_dir):
        print("prediction dir %s don't exist~"%pred_dir)
        exit(0)
    if not os.path.exists(gt_dir):
        print("gt dir %s don't exist~"%gt_dir)
        exit(0)
    metric_dicts = evaluate(pred_dir, gt_dir, transform, alg_name, dataset, mean=True)
    write_metric_to_file(alg_name+"-----%s"%dataset, metric_dicts, file_path)
  
