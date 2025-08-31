# Dreamark

### Environments

```
conda env create -f environment.yml
cd freqencoder
pip install .
cd ..
cd gridencoder
pip install .
cd ..
cd raymarching
pip install .
cd ..
```

### Generate Watermarked NeRF

```
./run.sh 0 "a pineapple."
```


The accuracy metrics are shown in 
```
./exp-dmtet-stage3/[experiment name]/log_df.txt
```
To calculate CLIP Score:
```
python eval_metrics.py
```

### Robustness evaluation

image-level attack
```
ckpt_path=[path to NeRF ckpt]
prompt=[prompt]

CUDA_VISIBLE_DEVICES=0 python main.py --text "$prompt" --scale 7.5 --dmtet --mesh_idx 0 --density_thresh 0.1 --finetune True --workspace exp-image/\
    --init_ckpt "$ckpt_path" \
    --test --trigger_size 1000 --msg_length 16 --eval_mark
```

model-level attack: Uncomment the arbitrary line in ```attack.py``` to evaluate finetuning or pruning
```
# trainer = Pruning(...
trainer = FTAL(...
```
Then run
```
ckpt_path=[path to NeRF ckpt]
prompt=[prompt]

CUDA_VISIBLE_DEVICES=0 python attack.py --text "$prompt"\
    --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 512  --w 512 --workspace exp-model/ --dmtet \
    --init_ckpt "$ckpt_path" \
    --finetune True --iters "40000" --eval_interval 1 --msg_length 16 --trigger_size 1000
```