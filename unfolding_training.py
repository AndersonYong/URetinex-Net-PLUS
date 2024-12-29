import argparse
import torch
import torch.nn as nn
import os 
import numpy as np
import random
from torch.utils.data import DataLoader
from data.load_patch import PatchLoading, EvalLoading
from data.preprocessing import InitailProcessing, Crop2Patch
from utils import save_TensorImg, write_config, write_imgs, write_loss, write_config_to_file, write_metric_to_file, save_np_inter
from model import UnfoldingModel
from torch.utils.tensorboard import SummaryWriter

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

def train(opts, epoch, dataloader, model, writer):
    print("Training the " + str(epoch) + " epoch ...")
    loss = {}
    iter_data_loader = iter(dataloader)
    for iteration, batch in enumerate(dataloader):
        patch_batch = make_patch(batch, opts)
        return_imgs, losses, lr = model(patch_batch)
        write_loss(writer=writer, loss=losses, epoch=iteration+len(iter_data_loader)*epoch)
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
            str_to_print += " lr : %0.7f" %lr
            print(str_to_print)
    return return_imgs, {l: loss[l] / float(iteration+1) for l in loss.keys()}

def val(opts, epoch, model, eval_dataloader):
    print("Evaling the " + str(epoch) + " epoch")
    metric = {}
    for iteration, batch in enumerate(eval_dataloader):
        metrics, results = model(batch, mode="eval")
        n_samples = results.size(0)
        #n_enhance_samples = enhance_results.size(0)
        if not os.path.exists(os.path.join(opts.saving_eval_dir, "img")):
            os.makedirs(os.path.join(opts.saving_eval_dir, "img"))
        save_TensorImg(results, path=os.path.join(opts.saving_eval_dir, "img", "image%d_epoch%d.png"%(iteration, epoch)), nrow=int(n_samples/4))

        for key in metrics.keys():
            if key not in metric.keys():
                metric[key] = metrics[key]
            else:
                metric[key] = metric[key] + metrics[key]
    print(" =========================================== > evaling done!")
    return {l: (metric[l] / float(iteration+1)) for l in metric.keys()}


def checkpoint(opts, model, model_path):
    state_dict = {
        "model_R": model.model_R.state_dict(),
        "model_L": model.model_L.state_dict(),
    }
        
    checkpoint_state = {
        "state_dict": state_dict,
        "optimizerG": model.optimizer_G.state_dict(),
        "epoch": model.epoch + 1,
        "opts": model.opts,
    }
    torch.save(checkpoint_state, model_path)
    print("saving to: ", model_path)

def create_suffix(opts):
    RLC_suffix = "IsPretrained-" + opts.second_stage + "-" + opts.R_model + "_R" + "-" + opts.L_model + "_L" + "--"
    one_step_model_suffix = "+t_" + str(opts.round) + "-Loss:[" + opts.loss_options + \
        "]+_gamma" + str(opts.gamma) + "_lamda" + str(opts.lamda) + "-"  + "offset_" + str(opts.Loffset)
    one_step_model_suffix += "--Ltv:" + str(opts.l_Ltv)
    model_suffix = RLC_suffix + one_step_model_suffix
    return one_step_model_suffix, model_suffix


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
    if not os.path.exists(opts.unfolding_model_dir):
        os.makedirs(opts.unfolding_model_dir)

    # ============== define save model path_suffix ====================
    one_step_model_suffix, model_suffix = create_suffix(opts)
    suffix = model_suffix + "--" + "pami---one_step_model.pth"
    model_path = os.path.join(opts.unfolding_model_dir, suffix)
    # ============== suffix define end
    opts.model_path = model_path
    print(" ")
    print("the current runing model: %s"%opts.model_path)

    # ============== define event files ==============================
    logdir_suffix = model_suffix
    writer = SummaryWriter(log_dir=os.path.join(opts.log_dir, logdir_suffix), filename_suffix=model_suffix)
    opts.saving_eval_dir = os.path.join(opts.saving_eval_dir, model_suffix)
    if not os.path.exists(opts.saving_eval_dir):
        os.makedirs(opts.saving_eval_dir)
    # write config to tensorboard
    write_config(writer, vars(opts))
    # =============== end

    write_config_to_file(vars(opts), os.path.join(opts.saving_eval_dir, "config.txt"))
    model.train()
    for epoch in range(opts.current_epoch, opts.epoch):
        model.epoch = epoch
        opts.current_epoch = epoch
        imgs, _ = train(opts, epoch=epoch, dataloader=train_dataloader, model=model, writer=writer)
        checkpoint(opts, model, model_path)
        if (epoch+1) % opts.eval_epoch == 0:
            checkpoint(opts, model, model_path + "ep%d" % (epoch+1))
        if (epoch+1) % opts.eval_epoch == 0:
            metric = val(opts, epoch+1, model, eval_dataloader)
            write_metric_to_file(metric, os.path.join(opts.saving_eval_dir, "metric.txt"), opts, epoch+1)
    writer.close()

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description='Configure')
    # training config
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--n_cpu', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--size', type=int, default=-1)
    parser.add_argument('--iteration_to_print', type=int, default=1)
    parser.add_argument('--current_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--Roffset', type=float, default=0.0)
    parser.add_argument('--Loffset', type=float, default=0.0)
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--freeze_decom', action="store_true")
    parser.add_argument('--second_stage', type=str, default="False")
    parser.add_argument('--concat_L', default=True, action='store_false')
    parser.add_argument('--gpu_id', type=str, default=1)
    
    # dataset dir
    parser.add_argument('--patch_low', type=str, default="")
    parser.add_argument('--patch_high', type=str, default="")

    parser.add_argument('--eval_low', type=str, default="")
    parser.add_argument('--eval_high', type=str, default="")
    parser.add_argument('--saving_eval_dir', type=str, default="")

    # math model penalty
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--lamda', type=float, default=1e-2)
    
    # model config
    parser.add_argument('--R_model', type=str, default="naive")
    parser.add_argument('--L_model', type=str, default="naive")
    parser.add_argument('--init', type=str, default="normal")
    parser.add_argument('--milestones', nargs='+', type=int)

    # loss function weights
    parser.add_argument('--loss_options', type=str, default="")
    parser.add_argument('--l_Pconstraint', type=float, default=0.0)
    parser.add_argument('--l_Qconstraint', type=float, default=0.0)
    parser.add_argument('--l_Ltv', type=float, default=0.0)
    # ----------------- loss arg for R ------------------------
    parser.add_argument('--l_R_l2', type=float, default=0.0)
    parser.add_argument('--l_R_ssim', type=float, default=0.0)
    parser.add_argument('--l_R_vgg', type=float, default=0.0)
    
    parser.add_argument('--unfolding_model_dir', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--Decom_model_low_path', type=str, default="")
    parser.add_argument('--Decom_model_high_path', type=str, default="")
    parser.add_argument('--pretrain_unfolding_model_path', type=str, default="")
    # log config
    parser.add_argument('--log_dir', type=str, default="/data/wengjian/low-light-enhancement/Ours/log/one_step_training")
    parser.add_argument('--write_imgs', type=int, default=5)
    opts = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    for k, v in vars(opts).items():
        print(k, v)
    
    model = UnfoldingModel(opts).cuda()
    run(opts, model)