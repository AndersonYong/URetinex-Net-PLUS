# URetinex-Net++
Official PyTorch implementation of URetinex-Net++: Interpretable Optimization-Inspired Unfolding Network for Low-light Image Enhancement.
The first version can be referred to [URetinex-Net](https://github.com/AndersonYong/URetinex-Net)

## Requirements
  1. Python == 3.7.6
  2. PyTorch == 1.4.0
  3. torchvision == 0.5.0

## Train
URetinex-Net++ is trained seperately for each module. First of all, you should train decomposition module for low-light images and normal-light images using the following script:

train decomposition for low-light
```
python decom_training.py --size 48 \
                            --epoch 2000 \
                            --batch_size 4 \
                            --eval_epoch 100 \
                            --n_cpu 1\
                            --saving_eval_dir "./eval_result/eval_decom/" \
                            --decom_model_dir "./model_ckpt/decom/" \
                            --patch_low "your low light training images" \
                            --patch_high "your nornal light training images" \
                            --eval_low "your low light eval images" \
                            --eval_high "your normal light eval images" \
                            --img_light "low" \
```
train decomposition for normal-light
```
python decom_training.py --size 48 \
                            --epoch 300 \
                            --batch_size 4 \
                            --eval_epoch 100 \
                            --n_cpu 1\
                            --saving_eval_dir "./eval_result/eval_decom/" \
                            --decom_model_dir "./model_ckpt/decom/" \
                            --patch_low "your low light training images" \
                            --patch_high "your nornal light training images" \
                            --eval_low "your low light eval images" \
                            --eval_high "your normal light eval images" \
                            --img_light "high" \
```

Then, you can train unfolding optimization module with pretrained decomposition network, this step would only optimize params for unfolding optimization module, and the decomposition network is fixed. Using the following script to train defaultly:
```
bash unfolding_training.sh
```
Or you could just specify your own hyper-params via 
```
python unfolding_training.py 
```
See `unfolding_training.py` to find the supported hyper-params.

Finally, train light and reflectance adjustment network using:
```
bash fusion_adjust_training.sh
```

## Evaluate
You can execute the following script to evaluate results on LOL eval datasets, the metric would be saved in ./evaluate_metric.txt. Run following sciript:
```
python test.py
```

Metric would exactly match the paper:
| MAE | SSIM | PSNR | LPIPS_COS | LPIPS | DISTS|
| :---: | :---: |:---: | :---: | :---: | :---: |
| 0.0589 | 0.8411 | 23.826 | 1.2115 | 0.2311 | 0.1015|



## Run single image
If you want to run single image, you may need to specify the enhance ratio, and input low-light image. Currently you may need to call `run_one_image` from `test.py` by your self, a more convinient interface would be supported in the future :)

## Citation
```
@article{wu2025interpretable,
  title={Interpretable Optimization-Inspired Unfolding Network for Low-Light Image Enhancement},
  author={Wu, Wenhui and Weng, Jian and Zhang, Pingping and Wang, Xu and Yang, Wenhan and Jiang, Jianmin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

***Noted that the code is only for non-commercial use! should you have any queries, contact me at***  wj1997s@163.com

