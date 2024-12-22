import torchvision
import torchvision.transforms as transforms
from torch.nn import init
import numpy as np
import os
import time
import torch
from PIL import Image
import glob
import cv2

def save_TensorImg(img_tensor, path, nrow=1):
    torchvision.utils.save_image(img_tensor, path, nrow=nrow)


def save_np_inter(data, path):
    for i in range(len(data)):
        data[i] = data[i].clone().cpu()
    [r_fs, l_fs, x1, x2, x3, x4, x5, n, R, R_gt] = data
    np.savez(path,  
        r_fs=r_fs,
        l_fs=l_fs,
        x1=x1,
        x2=x2, 
        x3=x3,
        x4=x4,
        x5=x5,
        n=n,
        R=R,
        R_gt=R_gt)

def np_save_TensorImg(img_tensor, path):
    img = np.squeeze(img_tensor.cpu().permute(0, 2, 3, 1).numpy())
    im = Image.fromarray(np.clip(img*255, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def write_loss(writer, loss, epoch):
    """ loss is a dict """
    writer.add_scalars('loss', loss, epoch)

def write_imgs(writer, imgs, epoch):
    for k, v in imgs.items():
        writer.add_images(k, v, epoch, dataformats="NCHW")

def write_config(writer, config):
    for k, v in config.items():
        writer.add_text(k, str(v), 0)
    
def gamma_correction(L, gamma):
    coef = 1 / gamma
    return L ** coef

def plot_hist(pred_R, gt_R):
    import cv2
    import matplotlib.pyplot as plt
    color = ('r', 'g', 'b')
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set(title='hist_predR')
    for i, col in enumerate(color):
        hist = cv2.calcHist([pred_R], [i], None, [100], [0, 1])
        ax1.plot(hist, color=col)
    ax2 = fig.add_subplot(122)
    ax2.set(title='hist_gtR')
    for i, col in enumerate(color):
        hist = cv2.calcHist([gt_R], [i], None, [100], [0, 1])
        ax2.plot(hist, color=col)
    return plt
    

def get_hist(pred_R, gt_R):
    pred_R = pred_R.squeeze(0).clone().cpu().permute(1, 2, 0).numpy()
    gt_R = gt_R.squeeze(0).clone().cpu().permute(1, 2, 0).numpy()
    plt = plot_hist(pred_R, gt_R)
    print("done")
    return plt


def split_and_unsqueeze(img_tensor):
    red_tensor = img_tensor[:,0,:,:].unsqueeze(1)
    green_tensor = img_tensor[:,1,:,:].unsqueeze(1)
    blue_tensor = img_tensor[:,2,:,:].unsqueeze(1)
    return [red_tensor, green_tensor, blue_tensor]

def write_config_to_file(config, file_path):
    """config should be a dict"""
    with open(file_path, 'w') as f:
        f.write(time.asctime(time.localtime(time.time())) + '\n')
        for key, value in config.items():
            f.write(key+"       "+str(value) + '\n')
        f.close()

def write_metric_to_file(metric, file_path, opts, epoch):
    """metric should be a dict, file_path should exist"""
    if epoch == opts.eval_epoch:
        with open(file_path, "w") as f:   
            f.close()
    with open(file_path, 'a') as f:
        f.write(time.asctime(time.localtime(time.time())) + '\n')
        f.write("epoch       " + str(epoch) + '\n')
        for key, value in metric.items():       
            f.write(key + "     "+str(value) + '\n')
        f.write('\n')
        f.close()
        

def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        m.weight.data.kaiming_normal_(a=0, mode='fan_in')
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.kaiming_normal_(a=0, mode='fan_in')
        m.bias.data.fill_(0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        init.xavier_normal_(m.weight.data, gain=0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)



def initial_model(model, opts):
    if opts.init == "normal":
        print("[*]------------------------normal initialization for model")
        model.apply(weights_init_normal)
    elif opts.init == "xavier":
        print("[*]------------------------xavier initialization for model")
        model.apply(weights_init_xavier)
    elif opts.init == "kaiming":
        print("[*]------------------------kaiming initialization for model")
        model.apply(weights_init_kaiming)
    else:
        print("init method not implemented, check utils.py")
        exit()
    return model

def define_modelR(opts):
    if opts.R_model == "HalfDnCNNSE":
        from network.restoration import HalfDnCNNSE
        model_R = HalfDnCNNSE(opts)
    else:
        print("model R not implemented")
        exit()
    model_R = initial_model(model_R, opts)
    return model_R

def define_modelL(opts):
    if opts.L_model == "Illumination_Alone":
        from network.illumination_enhance import Illumination_Alone
        model_L = Illumination_Alone(opts)
    else:
        print("model L not implemented")
        exit()
    model_L = initial_model(model_L, opts)
    return model_L

def define_compositor(opts):
    if opts.fusion_model == "weight3":
        from network.fusion_net import ESAFusion3
        model_fusion = ESAFusion3(opts)
    elif opts.fusion_model == "weight5":
        from network.fusion_net import ESAFusion5
        model_fusion = ESAFusion5(opts)
    elif opts.fusion_model == "weight7":
        from network.fusion_net import ESAFusion7
        model_fusion = ESAFusion7(opts)
    else:
        print("fusion model not implemented")
        exit()
    model_fusion = initial_model(model_fusion, opts)
    return model_fusion

def define_modelA(opts):
    if opts.A_model == "naive_res":
        from network.illumination_adjustment import Adjust_naive_res
        model_A = Adjust_naive_res(opts)
    else:
        print("model A not implemented")
        exit()
    model_A = initial_model(model_A, opts)
    return model_A

def imgs_for_each_t(t, return_imgs, I, P, Q, R, L):
    """
        input 'return_imgs' is a dict, 
        output also a dict
    """
    return_imgs["P"+str(t)] = P
    return_imgs["Q"+str(t)] = Q
    return_imgs["R"+str(t)] = R
    return_imgs["L"+str(t)] = L
    return_imgs["I.*P"+str(t)] = I * P
    return_imgs["P.*P"+str(t)] = P * P
    return return_imgs   

def print_loss(loss_dict, info, step):
    [epoch, iteration, len_dataloader] = info
    str_to_print = "Train_" + step + ": Epoch {}: {}/{} with ".format(
        epoch, iteration, len_dataloader
    )
    for key in loss_dict.keys():
        str_to_print += " %s : %0.4f | " % (key, loss_dict[key] / float(iteration+1))
    print(str_to_print)

def create_kernel_x(opts):
    if opts.grad_kernel == "sobel":
        filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        padding = 1
    elif opts.grad_kernel == "normal":
        filter = np.array([[0, 0], [-1, 1]], dtype=np.float32)
        padding = 0
    else:
        print("kernel not implemented yet")
        exit()
    return filter, filter.shape[0], padding

def create_kernel_y(opts):
    if opts.grad_kernel == "sobel":
        filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        padding = 1
    elif opts.grad_kernel == "normal":
        filter = np.array([[0, -1], [0, 1]], dtype=np.float32)
        padding = 0
    else:
        print("kernel not implemented yet")
        exit()
    return filter, filter.shape[0], padding

def load_param4Decom(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_decom = torch.load(decom_model_path)
        model.load_state_dict(checkpoint_decom['state_dict']['model_R'])
        print(" ******============>  loading pretrained Decomposition Low Model from: %s " % decom_model_path)
        # to freeze the params of Decomposition Model
        for param in model.parameters():
            param.required_grad = False

        return model
    else:
        print("pretrained Decomposition Model does not exist, check ---> %s " % decom_model_path)
        exit()

def load_decom(fusion_opts):
    def create_and_load(path):
        from network.decom import Decom
        model = Decom()
        if os.path.exists(path):
            ckpt = torch.load(path)
            model.load_state_dict(ckpt['state_dict']['model_R'])
            for param in model.parameters():
                param.requires_grad = False
            print(" ******============>  loading pretrained Decomposition Model from: %s " % path)
        else:
            print("pretrained Decomposition Model does not exist, check ---> %s " % path)
            exit()
        return model
    decom_low_model = create_and_load(fusion_opts.Decom_model_low_path)
    if ("net_L" not in fusion_opts) or ("net_L" in fusion_opts and fusion_opts.net_L == True) or "R_" in fusion_opts.adjust_L_loss:
        decom_high_model = create_and_load(fusion_opts.Decom_model_high_path)
    else:
        decom_high_model = None
    return decom_low_model, decom_high_model
    
def load_unfolding(opts):
    if os.path.exists(opts.pretrain_unfolding_model_path):
        checkpoint = torch.load(opts.pretrain_unfolding_model_path)
        old_opts = checkpoint["opts"]
        model_R = define_modelR(old_opts)
        model_L = define_modelL(old_opts)
        model_R.load_state_dict(checkpoint['state_dict']['model_R'])
        model_L.load_state_dict(checkpoint['state_dict']['model_L'])
        for param_R in model_R.parameters():
            param_R.requires_grad = False
        for param_L in model_L.parameters():
            param_L.requires_grad = False
        return old_opts, model_R, model_L
    else:
        print("pretrained R Model does not exist, check ---> %s"%opts.pretrain_unfolding_model_path)
        exit()

def load_AdjustFusion(opts):
    if os.path.exists(opts.old_model_A_path):
        checkpoint = torch.load(opts.old_model_A_path)
        AdjustFusion_opts = checkpoint["opts"]
        model_A = define_modelA(AdjustFusion_opts)
        model_A.load_state_dict(checkpoint['state_dict']['model_A'])
        for param_A in model_A.parameters():
            param_A.requires_grad = False
        if "fusion_model" in AdjustFusion_opts:
            model_fusion = define_compositor(AdjustFusion_opts)
            if model_fusion is not None:
                model_fusion.load_state_dict(checkpoint['state_dict']['model_compositor'])
                for param_fusion in model_fusion.parameters():
                    param_fusion.requires_grad = False
            return AdjustFusion_opts, model_A, model_fusion
        else:
            return AdjustFusion_opts, model_A, None        
    else:
        print("pretrained Adjust and Fusion Model does not exist, check --->")
        exit()


def random_split_dataset(ratio):
    import random
    low_path = "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_0.25/low_select/*.*"
    files_low = sorted(glob.glob(low_path))
    #files_high = sorted(glob.glob(high_path))
    len_files = len(files_low)
    selected = random.sample(list(np.arange(0, len_files)), 123)

    if not os.path.exists(os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_new_0.5", "low")):
        os.makedirs(os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_new_0.5", "low"))
    if not os.path.exists(os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_new_0.5", "high")):
        os.makedirs(os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_new_0.5", "high"))
    for idx in selected:
        low = files_low[idx]
        high = os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/high", os.path.basename(low))
        Image.open(low).save(os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_new_0.5/low", os.path.basename(low)))
        Image.open(high).save(os.path.join("/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/train_new_0.5/high", os.path.basename(high)))

        

def param_all(model,  net_input):
    import torchsummary
    shape = net_input.shape
    torchsummary.summary(model, (shape[1], shape[2], shape[3]))

def param_self_compute(model):
    parmas = 0
    for p in model.parameters():
        #print(p)
        parmas += p.numel()
    return parmas

def normalize_vis(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    return (tensor - min_value) / (max_value - min_value)

def connstuctSCIE():
    low_all = "/data/wengjian/low-light-enhancement/Ours/evaluate_data/Dataset_Part1"
    for_low = "/data/wengjian/low-light-enhancement/Ours/evaluate_data/low-source/SCIE"
    for_high = "/data/wengjian/low-light-enhancement/Ours/evaluate_data/real_high/SCIE_gt"
    if not os.path.exists(for_low):
        os.makedirs(for_low)
    if not os.path.exists(for_high):
       os.makedirs(for_high)
    random_idx = random.sample(range(1, 361), 120)
    for i in random_idx:
        low_img = Image.open(os.path.join(low_all, str(i), "2.JPG"))
        high_img = Image.open(os.path.join(low_all, "Label", "%d.JPG"%i))
        assert low_img.size[0] == high_img.size[0]
        assert low_img.size[1] == high_img.size[1]
        low_img = low_img.resize((int(low_img.size[0]*0.2), int(low_img.size[1]*0.2)), Image.ANTIALIAS)
        high_img = high_img.resize((int(high_img.size[0]*0.2), int(high_img.size[1]*0.2)), Image.ANTIALIAS)

        low_img.save(os.path.join(for_low, "%d.png"%i), quality=95)
        high_img.save(os.path.join(for_high, "%d.png"%i), quality=95)

def construct5K():
    low_all = "/data/wengjian/low-light-enhancement/fivek_dataset/pairs_photos/low_photos"
    for_low = "/data/wengjian/low-light-enhancement/pami/evaluate_data/low-source/5K"
    for_high = "/data/wengjian/low-light-enhancement/pami/evaluate_data/real_high/5K_gt"
    if not os.path.exists(for_low):
        os.makedirs(for_low)
    if not os.path.exists(for_high):
       os.makedirs(for_high)
    low_files = glob.glob(os.path.join(low_all, "*.*"))
    for file in low_files:
        name = os.path.basename(file).split('.')[0]
        print(name)
        low_img = Image.open(file)
        high_img = Image.open(os.path.join("/data/wengjian/low-light-enhancement/fivek_dataset/pairs_photos/ExpertE/png/%s.png"%name))
        print(low_img.size)
        print(high_img.size)
        assert low_img.size[0] == high_img.size[0]
        assert low_img.size[1] == high_img.size[1]
        low_img = low_img.resize((int(low_img.size[0]*0.1), int(low_img.size[1]*0.1)), Image.ANTIALIAS)
        high_img = high_img.resize((int(high_img.size[0]*0.1), int(high_img.size[1]*0.1)), Image.ANTIALIAS)

        low_img.save(os.path.join(for_low, "%s.png"%name), quality=95)
        high_img.save(os.path.join(for_high, "%s.png"%name), quality=95)

def feature_visualize(feat):
    feat = feat[0]
    c, w, h = feat.shape()
    feat_results = []
    for i in range(c):
        feat_results.append(feat[i:i+1].unsqueeze(0))
    return torch.cat(feat, dim=0)

def visualize_compositor(feat1, feat2, feat3):
    feat_1_results = feature_visualize(feat1)
    feat_2_results = feature_visualize(feat2)
    feat_3_results = feature_visualize(feat3)
    return feat_1_results, feat_2_results, feat_3_results

def see_lol():
    low_dir = "/data/wengjian/low-light-enhancement/pami/LOLdataset/train/low"
    file_list = sorted(glob.glob(low_dir+"/*.png"))
    transform = [
            transforms.ToTensor(),
    ]
    transforom = transforms.Compose(transform)
    grey_list = []
    for file in file_list:
        pil_img = Image.open(file)
        img_low = transforom(pil_img)
        grey_list.append(img_low.mean())
        #print(os.path.basename(file), img_low.mean())
    
    grey_list = np.array(grey_list)
    grey_list = torch.from_numpy(grey_list)
    mean = grey_list.mean()
    var = grey_list.var(unbiased=False)
    print(torch.sum(grey_list<mean-var))
    return mean, var

