CUDA_VISIBLE_DEVICES=6 & python train_mbae.py --debug 100 --batch_size 2 --num_workers 4 --dataset motionx --norm_first --dataset_type newjoints --amp --ema --steps_per_save_output 2500 --steps_per_log 200 --steps_per_checkpoint 10000 --motions_path /home/apujol/mbld/datasets/MotionX/MotionX/datasets/motion_data/new_joints --texts_path /home/apujol/mbld/datasets/MotionX/MotionX/datasets/texts/semantic_labels --log_dir logs/motionx_bae