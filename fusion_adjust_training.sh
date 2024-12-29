python adjust_L_training.py --size 96 \
                            --epoch 100 \
                            --batch_size 4 \
                            --fusion_model "weight3" \
                            --A_model "naive" \
                            --fusion_layers 1 2 3\
                            --eval_epoch 1 \
                            --gpu_id 3 --n_cpu 1\
                            --adjust_L_loss "rec-grad-spatial" --l_grad 1 --l_spa 10  \
                            --saving_eval_dir "./eval_result/eval_adjustment" \
                            --adjust_model_dir "./model_ckpt/adjust/" \
                            --eval_low "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/eval15/low" \
                            --eval_high "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/eval15/high"\
                            --patch_low "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/low" \
                            --patch_high "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/high" \
                            --milestones 12300 25000 60000 \
                            --min_ratio 1 \
                            --pretrain_unfolding_model_path "./pretrained_model/unfolding/unfolding_model.pth" \
                            --Decom_model_low_path './pretrained_model/decom/decom_low_light.pth' \
                            --Decom_model_high_path './pretrained_model/decom/decom_high_light.pth' \