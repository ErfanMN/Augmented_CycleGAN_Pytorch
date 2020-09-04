Aumented CycleGAN Based on this paper : 
  https://arxiv.org/pdf/1802.10151.pdf

And Implementation based on : 
  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


Here is an implementation of Augmented Cycle GAN in python 3 and pytorch 1.2 - [info : Will be updated very soon]

Steps for running on your custom data : 
</ 
np.save('Dataset/trainA.npy' , dataset[train_indices])
np.save('Dataset/trainB.npy' , labels[train_indices])
np.save('Dataset/valA.npy' , dataset[val_indices])
np.save('Dataset/valB.npy' , labels[val_indices])
>

The Try to run The following command to train your augmented model:
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot Dataset/ --name augcgan_mode

You can Use options to Change the training parameter and hyperparameters:
------------ Options -------------
batchSize: 32
beta1: 0.5
checkpoints_dir: ../drive/My Drive/augChecks
continue_train: False
dataroot: ../Dataset/
display_freq: 5000
enc_A_B: 1
epoch_count: 10
eval_A_freq: 1
eval_B_freq: 1
expr_dir: ../drive/My Drive/augChecks/augcgan_mode
gpu_ids: [0]
input_nc: 3
lambda_A: 1.0
lambda_B: 1.0
lambda_sup_A: 0.1
lambda_sup_B: 0.1
lambda_z_B: 0.025
lr: 0.0002
max_gnorm: 500.0
model: aug_cycle_gan
monitor_gnorm: True
name: augcgan_mode
ndf: 64
nef: 32
ngf: 32
niter: 25
niter_decay: 25
nlatent: 16
no_lsgan: False
norm: instance
num_multi: 10
numpy_data: 1
output_nc: 1
print_freq: 100
save_epoch_freq: 1
seed: None
stoch_enc: False
sup_frac: 0.1
supervised: False
use_dropout: False
which_epoch: latest
which_model_netD: basic
which_model_netG: resnet
z_gan: 1


Note : to change any of the options you can add the option to train command line from above by -- 
example --output_nc 10
