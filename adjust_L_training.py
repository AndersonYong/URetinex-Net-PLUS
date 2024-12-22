from ast import Store
from operator import irshift
from wsgiref.simple_server import demo_app
import torch
import torch.nn as nn
import os 
import numpy as np
import random
from torch.utils.data import DataLoader
from data.load_patch import PatchLoading, EvalLoading
from data.preprocessing import InitailProcessing, Crop2Patch
from utils import save_TensorImg, write_config, write_loss, write_config_to_file, write_imgs, write_metric_to_file, np_save_TensorImg
from model import AdjustModel
import argparse
from data.data_augmentation import augmentation

def make_patch(batch, opts):
    patch_batch = {}
    input_low_img = batch["low_light_img"]
    ref_high_img = batch["high_light_img"]
    bs, c, h, w = input_low_img.shape
    patch_low = torch.zeros((bs, c, opts.size, opts.size), dtype=torch.float32)
    patch_high = torch.zeros((bs, c, opts.size, opts.size), dtype=torch.float32)
    for i in range(bs):
        # random choose a idx to crop a patch
        x = random.randint(0, h-opts.size)
        y = random.randint(0, w-opts.size)
        patch_low[i, :, :, :] = input_low_img[i, :, x: x+opts.size, y: y+opts.size]
        patch_high[i, :, :, :] = ref_high_img[i, :, x: x+opts.size, y:y+opts.size]
    patch_batch['low_light_img'] = patch_low
    patch_batch['high_light_img'] = patch_high
    return patch_batch

def train(opts, epoch, dataloader, model):
    print("Training the " + str(epoch) + " epoch ...")
    loss = {}
    for iteration, batch in enumerate(dataloader):
        patch_batch = make_patch(batch, opts)
        losses, lr = model(patch_batch)
        for key in losses.keys():
            if key in loss.keys():
                loss[key] = losses[key].cpu().mean().detach().item() + loss[key]
            else:
                loss[key] = losses[key].cpu().mean().detach().item()
        if iteration % opts.iteration_to_print == 0:
            str_to_print = "Train: Epoch {}: {}/{} with ".format(
                epoch, iteration, len(dataloader)
            )
            for key in loss.keys():
                str_to_print += " %s : %0.6f | " % (key, loss[key] / float(iteration+1))
            str_to_print += " %s : %0.6f | " % ("lr", lr)
            print(str_to_print)
        
    return {l: loss[l] / float(iteration) for l in loss.keys()}

def checkpoint(model, model_path):
    state_dict = {
        "model_decom": model.model_Decom_low.state_dict(),
        "model_R": model.model_R.state_dict(),
        "model_L": model.model_L.state_dict(),
        "model_A": model.adjust_model.state_dict(),
    }
    if model.fusion_model is not None:
        state_dict["model_compositor"] = model.fusion_model.state_dict()
    checkpoint_state = {
        'epoch': model.epoch + 1,
        'state_dict': state_dict,
        'opts': model.opts,
        'optimizer':{
            "model_A": model.optimizer_A.state_dict()
        }
    }
    torch.save(checkpoint_state, model_path)

def val(opts, epoch, model, eval_dataloader):
    print("Evaling the " + str(epoch) + " epoch")
    metric = {}
    for iteration, batch in enumerate(eval_dataloader):
        metrics, enhance_results = model(batch, mode="eval")
        if not os.path.exists(os.path.join(opts.saving_eval_dir, "img")):
            os.makedirs(os.path.join(opts.saving_eval_dir, "img"))
        save_TensorImg(enhance_results, path=os.path.join(opts.saving_eval_dir, "img", "image%d_enhance_epoch%d.png"%(iteration, epoch)), nrow=int(enhance_results.size(0)))
        
        for key in metrics.keys():
            if key not in metric.keys():
                metric[key] = metrics[key]
            else:
                metric[key] = metric[key] + metrics[key]
    print(" =========================================== > evaling done!")
    return {l: (metric[l] / float(iteration+1)) for l in metric.keys()}

def create_suffix(opts):
    t_stage = os.path.basename(opts.pretrain_unfolding_model_path).split('--A_t_')[1].split('_')[0]
    RLC_suffix = "layers=" + str(opts.fusion_layers) + "-stage=%s-"%t_stage + "size=%d-"%opts.size + \
        "batch=%d-"%opts.batch_size + opts.A_model + "_A" + "--" +  opts.fusion_model + "_fusion" + "-NetHigh:" + str(opts.net_L)
    one_step_model_suffix = "-Loss:[" + opts.adjust_L_loss +  "]-weights:[l_grad:%0.1f-l_spa:%0.2f"%(opts.l_grad, opts.l_spa) + "_min_ratio:%0.2f"%opts.min_ratio + "]"
    model_suffix = RLC_suffix + one_step_model_suffix
    return model_suffix

def reload(model, path):
    ckpt = torch.load(path)
    model.adjust_model.load_state_dict(ckpt["state_dict"]["model_A"])
    if model.fusion_model is not None:
        model.fusion_model.load_state_dict(ckpt["state_dict"]["model_compositor"])
    return ckpt["epoch"]

def run(opts, model):
    train_dataset = PatchLoading(opts)
    train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=opts.batch_size,
                    shuffle=True,
                    drop_last = True,
                    num_workers=opts.n_cpu)
    
    eval_dataset = EvalLoading(opts)
    eval_dataloader = DataLoader(
                    eval_dataset, 
                    batch_size=1,
                    shuffle=False,
                    drop_last = False,
                    num_workers=opts.n_cpu)
    print(len(iter(train_dataloader)))
    print(len(iter(eval_dataloader)))

    if not os.path.exists(opts.adjust_model_dir):
        os.makedirs(opts.adjust_model_dir)
    model_suffix = create_suffix(opts)
    opts.model_path = os.path.join(opts.adjust_model_dir, model_suffix)

    opts.saving_eval_dir = os.path.join(opts.saving_eval_dir, model_suffix)
    if not os.path.exists(opts.saving_eval_dir):
        os.makedirs(opts.saving_eval_dir)

    write_config_to_file(vars(opts), os.path.join(opts.saving_eval_dir, "config.txt"))

    model.train() 
    continue_epoch = 0
    if opts.reload:
        if os.path.exists(opts.model_path):
            continue_epoch = reload(model, opts.model_path)
            print("----------------------------------------------------reloading epoch: %d----------------------------------------------------"%continue_epoch)
    else:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  do not reload  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for epoch in range(continue_epoch, opts.epoch):
        model.epoch = epoch
        train_loss = train(opts, epoch=epoch, dataloader=train_dataloader, model=model)
        checkpoint(model, opts.model_path+".pth")
        if (epoch+1) % opts.eval_epoch == 0:
            checkpoint(model, opts.model_path + "ep%d.pth" % (epoch+1))
        if (epoch+1) % opts.eval_epoch == 0:
            metric = val(opts, epoch+1, model, eval_dataloader)
            write_metric_to_file(metric, os.path.join(opts.saving_eval_dir, "metric.txt"), opts, epoch+1)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    # training config
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_cpu', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--iteration_to_print', type=int, default=1)
    parser.add_argument('--current_epoch', type=int, default=0)
    parser.add_argument('--eval_epoch', type=int, default=None)
    parser.add_argument('--fusion_layers', nargs='+', type=int, default=None)
    parser.add_argument('--net_L', action="store_false")
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--milestones', nargs='+', type=int, default=[None])
    parser.add_argument('--min_ratio', type=float, default=None)

    # dataset dir
    parser.add_argument('--patch_low', type=str, default=None)
    parser.add_argument('--patch_high', type=str, default=None)

    parser.add_argument('--eval_low', type=str, default=None)
    parser.add_argument('--eval_high', type=str, default=None)
    parser.add_argument('--saving_eval_dir', type=str, default=None)

    # loss config
    parser.add_argument('--l_grad', type=float, default=0)
    parser.add_argument('--l_spa', type=float, default=0)
    parser.add_argument('--adjust_L_loss', type=str, default=None)
    # model config
    parser.add_argument('--A_model', type=str, default=None)
    parser.add_argument('--fusion_model', type=str, default=None)
    parser.add_argument('--init', type=str, default="normal")
    parser.add_argument('--reload', action="store_false")

    # checkpoints config
    parser.add_argument('--adjust_model_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--Decom_model_low_path', type=str, default="/data/wengjian/low-light-enhancement/pami/pretrained_model/decom4layers/decom-L_supervised-4layers")
    parser.add_argument('--Decom_model_high_path', type=str, default="/data/wengjian/low-light-enhancement/pami/pretrained_model/decom4layers/decom_onlyHigh_0.1L_2.2gamma_0.1Lawareep340")
    parser.add_argument('--pretrain_unfolding_model_path', type=str, default=None)

    opts = parser.parse_args()

    # print all the parameters
    for k, v in vars(opts).items():
        print(k, v)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    model = AdjustModel(opts).cuda()
    run(opts, model=model)
    