python decom_training.py --size 48 \
                            --epoch 20 \
                            --batch_size 4 \
                            --eval_epoch 10 \
                            --n_cpu 1\
                            --saving_eval_dir "./eval_result/eval_decom/" \
                            --decom_model_dir "./model_ckpt/decom/" \
                            --patch_low "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/low" \
                            --patch_high "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/high" \
                            --eval_low "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/eval15/low" \
                            --eval_high "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/eval15/high" \
                            --img_light "high" \
                            #--reload
