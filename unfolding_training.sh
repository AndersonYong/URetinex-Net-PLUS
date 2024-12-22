python unfolding_training.py --R_model 'HalfDnCNNSE' --gamma 0.5 \
                            --L_model 'Illumination_Alone' --lamda 0.5 \
                            --loss_options 'RL2(1.0)-RSSIM(1.0)-RVGG(1.0)-Ltv(20)' \
                            --round 3 --batch_size 4 --init 'normal' --epoch 2000\
                            --size 48 --eval_epoch 2 \
                            --lr 2e-4 --write_imgs 10 \
                            --l_R_l2 1.0  --l_R_vgg 1.0  --l_R_ssim 1.0  --l_Ltv 20 \
                            --l_Pconstraint 1 --l_Qconstraint 1\
                            --Loffset 0.05 --Roffset 0.05\
                            --milestones 30000  --freeze_decom --second_stage "False"\
                            --Decom_model_low_path '/data/wengjian/low-light-enhancement/Ours/pretrained_model/decom/decom-L_supervised-4layers' \
                            --Decom_model_high_path '/data/wengjian/low-light-enhancement/Ours/pretrained_model/decom/decom4layers/decom_onlyHigh_0.1L_2.2gamma_0.1Lawareep340' \
                            --unfolding_model_dir  "./model_ckpt/unfolding/"\
                            --saving_eval_dir "./eval_result/eval_unfolding/"\
                            --patch_low "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/low" \
                            --patch_high "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/our485/high" \
                            --eval_low "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/eval15/low" \
                            --eval_high "/data/wengjian/low-light-enhancement/Ours/dataset/LOLdataset/eval15/high" \
                            --log_dir "./log/unfolding_training" \
                            --gpu_id 1 \
                            --pretrain_unfolding_model_path ''
                            #--concat_L \ 
                            


                    

 
