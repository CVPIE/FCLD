# FCLD

Official PyTorch implementation of **FCLD: Unsupervised Medical Image Translation with Frequency-Cross Latent Diffusion Model**.

## Dataset
You should structure your aligned dataset in the following way:

```
INPUT_PATH/
  ├── trainA
        ├──***.png
  ├── trainB
  ├── valA
  ├── ValB
  ├── testA
  ├── testB
```

## Train

<br />

```
cd FCLD
python3 train_DDIM.py --image_size 256 --latent_size 32 --exp exp --num_channels 8 --num_timesteps 1000 --batch_size 2 --num_epoch 500 --use_ema --ema_decay 0.999 --lr_d 1e-4 --lr_g 1.5e-4 --num_process_per_node 1 --save_content --local_rank 0 --input_path INPUT_PATH --output_path results --net_type unet --AE_ckpt_path  AE_PATH

```

<br />

## Test

<br />

```
cd FCLD
python test_DDIM.py --image_size 256 --latent_size 32 --exp exp --num_channels 8 --num_timesteps 1000 --ddim_num_steps 200 --batch_size 64 --which_epoch 350 --phase test --gpu_chose 0 --input_path INPUT_PATH --output_path results --net_type unet --AE_ckpt_path AE_ckpt_path --b2a --a2b --save_img
```

<br />

