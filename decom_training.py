import torch
import torch.nn as nn
import os 
import numpy as np
import random
from torch.utils.data import DataLoader
from data.load_patch import PatchLoading, EvalLoading
from data.preprocessing import InitailProcessing, Crop2Patch
from utils import save_TensorImg, write_config, write_loss, write_config_to_file, write_imgs
from model import DecomModel
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def make_patch(batch, opts):
    patch_batch = {}
    input_low_img = batch["low_light_img"]
    ref_high_img = batch["high_light_img"]
    bs, c, h, w = input_low_img.shape
    patch_low = torch.zeros((bs, c, opts.size, opts.size), dtype=torch.float32).to(input_low_img.device)
    patch_high = torch.zeros((bs, c, opts.size, opts.size), dtype=torch.float32).to(input_low_img.device)
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
    iter_data_loader = iter(dataloader)
    for iteration, batch in enumerate(dataloader):
        patch_batch = make_patch(batch, opts)
        losses = model(patch_batch)
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
            print(str_to_print)
        
    return {l: loss[l] / float(iteration) for l in loss.keys()}

def checkpoint(model, model_path):
    checkpoint_state = {
        'epoch': model.epoch + 1,
        'state_dict': {
            "model_R": model.decomModel.state_dict()
        },
        'optimizer':{
            "model_R": model.optimizer_D.state_dict()
        }
    }
    torch.save(checkpoint_state, model_path)


def val(opts, epoch, model, eval_dataloader):
    print("Evaling the " + str(epoch) + " epoch")
    for iteration, batch in enumerate(eval_dataloader):
        results = model(batch, mode="eval")
        n_results = results.size(0)
        save_TensorImg(results, path=os.path.join(opts.saving_eval_dir, "image%d_epoch%d.png"%(iteration, epoch)), nrow=n_results)
    print("evaling done!")

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
    assert (opts.img_light == "low" or opts.img_light == "high")
    
    if not os.path.exists(os.path.join(opts.decom_model_dir, opts.img_light)):
        os.makedirs(os.path.join(opts.decom_model_dir, opts.img_light))
    opts.model_path = os.path.join(opts.decom_model_dir, opts.img_light, "decom_model.pth")

    opts.saving_eval_dir = os.path.join(opts.saving_eval_dir, opts.img_light)
    if not os.path.exists(opts.saving_eval_dir):
        os.makedirs(opts.saving_eval_dir)

    model.train() 
    for epoch in range(0, opts.epoch):
        model.epoch = epoch
        train_loss = train(opts, epoch=epoch, dataloader=train_dataloader, model=model)
        checkpoint(model, opts.model_path)
        if (epoch+1) % opts.eval_epoch == 0:
            checkpoint(model, opts.model_path + "ep%d" % (epoch+1))
        if (epoch+1) % opts.eval_epoch == 0:
            val(opts, epoch+1, model, eval_dataloader)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training config
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_cpu', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--size', type=int, default=48)
    parser.add_argument('--iteration_to_print', type=int, default=1)
    parser.add_argument('--eval_epoch', type=int, default=20)
    parser.add_argument('--img_light', type=str, default="")

    # dataset dir
    parser.add_argument('--patch_low', type=str, default="")
    parser.add_argument('--patch_high', type=str, default="")
    parser.add_argument('--eval_low', type=str, default="")
    parser.add_argument('--eval_high', type=str, default="")
    parser.add_argument('--saving_eval_dir', type=str, default="")

    # model config
    parser.add_argument('--init', type=str, default="normal")

    # checkpoints config
    parser.add_argument('--decom_model_dir', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")

    opts = parser.parse_args()

    # print all the parameters
    for k, v in vars(opts).items():
        print(k, v)

    decom = DecomModel(opts).cuda()
    run(opts, model=decom)
    
