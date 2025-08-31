
# stage 1
CUDA_VISIBLE_DEVICES=5 python main.py --text "a pineapple."\
    --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 512  --w 512 --workspace exp-nerf-stage1/

# stage 2
CUDA_VISIBLE_DEVICES=0 python main.py --text "a pineapple."\
    --iters 15000 --scale 100 --dmtet --mesh_idx 0  \
    --init_ckpt exp-nerf-stage1/2024-07-11-a-pineapple.-Uchida-scale-7.5-lr-0.001-albedo-le-10.0-render-512-cube-sd-2.1-5000-tet-256/checkpoints/best_df_ep0250.pth\
    --normal True --sds True --density_thresh 0.1 --lambda_normal 5000 --workspace exp-dmtet-stage2/

# stage 3
CUDA_VISIBLE_DEVICES=0 python main.py --text "a pineapple."\
    --iters 30000 --scale 7.5 --dmtet --mesh_idx 0  \
    --init_ckpt exp-dmtet-stage2/2024-07-13-a-pineapple.-scale-100.0-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-sds-normal-tet-256-lnorm-5000.0/checkpoints/best_df_ep0150.pth\
    --density_thresh 0.1 --finetune True --workspace exp-dmtet-stage3/ --eval_mark

# watermark
CUDA_VISIBLE_DEVICES=6 python main.py --text "A rotrary telephone carved out of wood."\
    --iters 30000 --scale 7.5 --dmtet --mesh_idx 0  \
    --init_ckpt exp-dmtet-stage2/2024-08-19-A-rotary-telephone-carved-out-of-wood.-scale-100.0-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-sds-normal-tet-256-lnorm-5000.0/checkpoints/best_df_ep0150.pth\
    --density_thresh 0.1 --finetune True --workspace exp-dmtet-stage3/ --msg_length 16 --eval_mark

# finetune
CUDA_VISIBLE_DEVICES=3 python attack.py --text "a DSLR photo of peacock."\
    --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 512  --w 512 --workspace exp-finetune/ --dmtet \
    --init_ckpt "exp-dmtet-stage3/2024-07-26-a-DSLR-photo-of-peacock.-16-1000-scale-7.5-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-finetune-tet-256/checkpoints/best_df_ep0300.pth" \
    --finetune True --iters "40000" --eval_interval 1 --msg_length 16 --trigger_size 1000

# pruning
CUDA_VISIBLE_DEVICES=3 python attack.py --text "a DSLR photo of peacock."\
    --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 512  --w 512 --workspace exp-pruning/ --dmtet \
    --init_ckpt "exp-dmtet-stage3/2024-07-26-a-DSLR-photo-of-peacock.-16-1000-scale-7.5-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-finetune-tet-256/checkpoints/best_df_ep0300.pth" \
    --finetune True --iters "40000" --eval_interval 1 --msg_length 16 --trigger_size 1

# test
CUDA_VISIBLE_DEVICES=1 python main.py --text "a DSLR photo of peacock." --scale 7.5 --dmtet --mesh_idx 0 --density_thresh 0.1 --finetune True --workspace exp-test/\
    --init_ckpt "exp-dmtet-stage3/2024-07-26-a-DSLR-photo-of-peacock.-16-1000-scale-7.5-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-finetune-tet-256/checkpoints/best_df_ep0300.pth" \
    --test --trigger_size 1000 --msg_length 16 --eval_mark
