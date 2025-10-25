

import argparse
import torch
import numpy as np
import time

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dataset import CreateDatasetSynthesis

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_msssim import ssim, ms_ssim

import random
np.random.seed(11)
random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

dwt = DWT_2D("haar")
iwt = IDWT_2D("haar")

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

def DWT_func(x, layer=1):
    y = x
    for i in range(layer):
        xll, xlh, xhl, xhh = dwt(y)
        y = torch.cat([xll, xlh, xhl, xhh], dim=1)
        y = y / 2.0
    return y

#%%
def train_syndiff(rank, gpu, args):

    from backbones.autoencoder import AutoencoderKL

    from backbones.unet import UNetModel
    
    from backbones.ddim import DDIMSampler
    
    import backbones.generator_resnet 
    
    
    from utils.EMA import EMA
    
    #rank = args.node_rank * args.num_process_per_node + gpu
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    dataset = CreateDatasetSynthesis("train", args)
    dataset_val = CreateDatasetSynthesis("val", args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                               batch_size=8,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=val_sampler,
                                               drop_last = True)

    print('train dataloader size:'+str(len(data_loader)))
    print('val dataloader size:'+str(len(data_loader_val)))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    G_NET_ZOO = {"unet": UNetModel}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))

    # load AE
    AE = AutoencoderKL(embed_dim=4,double_z=True,z_channels=4,in_channels=1,out_ch=1)
    AE.init_from_ckpt(args.AE_ckpt_path, ['loss'])
    AE = AE.eval().to(device)
    for param in AE.parameters():
        param.requires_grad = False

    tome_info = None
    args.image_size = args.latent_size
    concat_channels = args.num_channels

    gen_diffusive_1 = gen_net(image_size=args.latent_size,in_channels=concat_channels,out_channels=concat_channels,model_channels=192,
                                attention_resolutions=[1,2,4,8],num_res_blocks=2,channel_mult=[1,2,2,4,4],
                                num_heads=8,use_scale_shift_norm=True,resblock_updown=True,
                                use_spatial_transformer=True, use_spatial_transformer_bottlenet=True,
                                use_VRKWV=True, RKWV_resolutions=[1, 2],
                                context_dim=16,context_size=(64, 64),tome_info=tome_info).to(device)
    gen_diffusive_2 = gen_net(image_size=args.latent_size,in_channels=concat_channels,out_channels=concat_channels,model_channels=192,
                                attention_resolutions=[1,2,4,8],num_res_blocks=2,channel_mult=[1,2,2,4,4],
                                num_heads=8,use_scale_shift_norm=True,resblock_updown=True,
                                use_spatial_transformer=True, use_spatial_transformer_bottlenet=True,
                                use_VRKWV=True, RKWV_resolutions=[1, 2],
                                context_dim=16,context_size=(64, 64),tome_info=tome_info).to(device)

    #networks performing translation
    args.num_channels = args.num_channels // 2
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(input_nc=1, output_nc=1, netG='resnet_6blocks', gpu_ids=[gpu])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(input_nc=1, output_nc=1, netG='resnet_6blocks', gpu_ids=[gpu])
    
    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(input_nc=1, gpu_ids=[gpu])
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(input_nc=1, gpu_ids=[gpu])
    
    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())
    broadcast_params(gen_non_diffusive_1to2.parameters())
    broadcast_params(gen_non_diffusive_2to1.parameters())
    broadcast_params(disc_non_diffusive_cycle1.parameters())
    broadcast_params(disc_non_diffusive_cycle2.parameters())
    
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    optimizer_gen_non_diffusive_1to2 = optim.Adam(gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))

    optimizer_disc_non_diffusive_cycle1 = optim.Adam(disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))    


    optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
    optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)
    optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=args.ema_decay)
    optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=args.ema_decay)
        
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_1to2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_2to1, args.num_epoch, eta_min=1e-5)    
    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle2, args.num_epoch, eta_min=1e-5)
    
    #ddp
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])
    gen_non_diffusive_1to2 = nn.parallel.DistributedDataParallel(gen_non_diffusive_1to2, device_ids=[gpu])
    gen_non_diffusive_2to1 = nn.parallel.DistributedDataParallel(gen_non_diffusive_2to1, device_ids=[gpu])    
    disc_non_diffusive_cycle1 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle1, device_ids=[gpu])
    disc_non_diffusive_cycle2 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle2, device_ids=[gpu])
    
    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path,exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        copy_source(__file__, exp_path)
        target_dir = os.path.join(exp_path, 'backbones')
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree('./backbones', target_dir)
    
    # ddim sampler
    sampler = DDIMSampler(device=device, ddim_num_steps=200, ddim_eta=1., verbose=True, beta_schedule="linear", timesteps=args.num_timesteps)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        gen_non_diffusive_1to2.load_state_dict(checkpoint['gen_non_diffusive_1to2_dict'])
        gen_non_diffusive_2to1.load_state_dict(checkpoint['gen_non_diffusive_2to1_dict'])        
        # load G
        
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2']) 
        optimizer_gen_non_diffusive_1to2.load_state_dict(checkpoint['optimizer_gen_non_diffusive_1to2'])
        scheduler_gen_non_diffusive_1to2.load_state_dict(checkpoint['scheduler_gen_non_diffusive_1to2'])
        optimizer_gen_non_diffusive_2to1.load_state_dict(checkpoint['optimizer_gen_non_diffusive_2to1'])
        scheduler_gen_non_diffusive_2to1.load_state_dict(checkpoint['scheduler_gen_non_diffusive_2to1'])           
        # load D_for cycle
        if not args.no_use_disc:
            disc_non_diffusive_cycle1.load_state_dict(checkpoint['disc_non_diffusive_cycle1_dict'])
            optimizer_disc_non_diffusive_cycle1.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle1'])
            scheduler_disc_non_diffusive_cycle1.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle1'])

            disc_non_diffusive_cycle2.load_state_dict(checkpoint['disc_non_diffusive_cycle2_dict'])
            optimizer_disc_non_diffusive_cycle2.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle2'])
            scheduler_disc_non_diffusive_cycle2.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle2'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    names, _, x_val , y_val = next(iter(data_loader_val))

    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
       
        for iteration, (x1, x2) in enumerate(data_loader):

            start_time = time.time()

            #sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            # wavelet condition
            cond_data_1 = DWT_func(real_data1, 2)
            cond_data_2 = DWT_func(real_data2, 2)

            # encode
            real_data1 = AE.get_latent(real_data1)
            real_data2 = AE.get_latent(real_data2)

            # data for GAN
            real_data1_non_diffusive = x1.to(device, non_blocking=True)
            real_data2_non_diffusive = x2.to(device, non_blocking=True)

            for p in disc_non_diffusive_cycle1.parameters():  
                p.requires_grad = True  
            for p in disc_non_diffusive_cycle2.parameters():  
                p.requires_grad = True   

            #D for cycle part
            disc_non_diffusive_cycle1.zero_grad()
            disc_non_diffusive_cycle2.zero_grad()

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1_non_diffusive).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2_non_diffusive).view(-1) 
            
            errD_cycle1_real = F.softplus(-D_cycle1_real)
            errD_cycle1_real = errD_cycle1_real.mean()            
            
            errD_cycle2_real = F.softplus(-D_cycle2_real)
            errD_cycle2_real = errD_cycle2_real.mean()   
            errD_cycle_real = errD_cycle1_real + errD_cycle2_real
            errD_cycle_real.backward(retain_graph=True)
            # train with fake
            
            x1_0_predict = gen_non_diffusive_2to1(real_data2_non_diffusive)
            x2_0_predict = gen_non_diffusive_1to2(real_data1_non_diffusive)

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1) 
            
            errD_cycle1_fake = F.softplus(D_cycle1_fake)
            errD_cycle1_fake = errD_cycle1_fake.mean()            
            
            errD_cycle2_fake = F.softplus(D_cycle2_fake)
            errD_cycle2_fake = errD_cycle2_fake.mean()   
            errD_cycle_fake = errD_cycle1_fake + errD_cycle2_fake
            errD_cycle_fake.backward()

            errD_cycle = errD_cycle_real + errD_cycle_fake
            # Update D
            optimizer_disc_non_diffusive_cycle1.step()
            optimizer_disc_non_diffusive_cycle2.step() 

            #G part
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = False                
            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad()
            gen_non_diffusive_2to1.zero_grad()   
            
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            
            #sample x_t
            noise1 = torch.randn_like(real_data1)
            noise2 = torch.randn_like(real_data2)         
            x1_t = q_sample(real_data1, t1, noise1, sampler.sqrt_alphas_cumprod, sampler.sqrt_one_minus_alphas_cumprod)   
            x2_t = q_sample(real_data2, t2, noise2, sampler.sqrt_alphas_cumprod, sampler.sqrt_one_minus_alphas_cumprod)             
            
            #translation networks
            x1_0_predict = gen_non_diffusive_2to1(real_data2_non_diffusive)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1_non_diffusive)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)


            #x_tp1 is concatenated with source contrast and x_0_predict is predicted
            eps_x1 = gen_diffusive_1(torch.cat((x1_t.detach(),AE.get_latent(x2_0_predict)),axis=1), t1, DWT_func(x2_0_predict, 2))
            eps_x2 = gen_diffusive_2(torch.cat((x2_t.detach(),AE.get_latent(x1_0_predict)),axis=1), t2, DWT_func(x1_0_predict, 2))  

            #D_cycle output for fake x1_0_predict
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1) 
        
            errG_cycle_adv1 = F.softplus(-D_cycle1_fake)
            errG_cycle_adv1 = errG_cycle_adv1.mean()            
            
            errG_cycle_adv2 = F.softplus(-D_cycle2_fake)
            errG_cycle_adv2 = errG_cycle_adv2.mean()   
            errG_cycle_adv = errG_cycle_adv1 + errG_cycle_adv2
            
            #L1 loss 
            errG1_L1 = F.l1_loss(eps_x1[:,:args.num_channels,:],noise1)
            errG2_L1 = F.l1_loss(eps_x2[:,:args.num_channels,:],noise2)
            errG_L1 = errG1_L1 + errG2_L1
            
            #cycle loss
            errG1_cycle=F.l1_loss(x1_0_predict_cycle,real_data1_non_diffusive)
            errG2_cycle=F.l1_loss(x2_0_predict_cycle,real_data2_non_diffusive)
            errG_cycle = errG1_cycle + errG2_cycle 

            torch.autograd.set_detect_anomaly(True)
            
            errG = args.lambda_l1_loss*errG_cycle + errG_cycle_adv + args.lambda_l1_loss*errG_L1
            errG.backward()
            
            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step()
            optimizer_gen_non_diffusive_2to1.step()         
            
            global_step += 1
            if iteration % 10 == 0:
                if rank == 0:
                    print('time per iteration: {}'.format(time.time() - start_time))
                    print('epoch {} iteration{}, G-Cycle: {}, G-L1: {}, G-cycle-Adv: {}, G-Sum: {}, D_cycle Loss: {}'.format(epoch,iteration, errG_cycle.item(), errG_L1.item(), errG_cycle_adv.item(), errG.item(), errD_cycle.item()))
        
        if not args.no_lr_decay:
            
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()

            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()

        if args.save_content:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'gen_diffusive_1_dict': gen_diffusive_1.state_dict(), 'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                            'gen_diffusive_2_dict': gen_diffusive_2.state_dict(), 'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                            'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(),
                            'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(),
                            'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(), 'optimizer_gen_non_diffusive_1to2': optimizer_gen_non_diffusive_1to2.state_dict(),
                            'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(), 'optimizer_gen_non_diffusive_2to1': optimizer_gen_non_diffusive_2to1.state_dict(),
                            'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(), 'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                            'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(), 'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                            'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(), 'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
                            'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict()}
                torch.save(content, os.path.join(exp_path, 'content.pth'))
        
        if rank == 0:

            with torch.no_grad():
                rec_img1 = AE.decode(real_data1)
                rec_img1 = (torch.clamp(rec_img1, -1, 1) + 1) / 2
                real_img1 = (torch.clamp(real_data1_non_diffusive, -1, 1) + 1) / 2

                rec_img2 = AE.decode(real_data2)
                rec_img2 = (torch.clamp(rec_img2, -1, 1) + 1) / 2
                real_img2 = (torch.clamp(real_data2_non_diffusive, -1, 1) + 1) / 2

                #concatenate noise and source contrast
                fake_sample1, _ = sampler.sample(model=gen_diffusive_1, batch_size=real_data2.shape[0], shape=real_data2.shape[1:], 
                                                 conditioning=real_data2, cond_for_CrossAttn=cond_data_2)
                fake_sample1 = AE.decode(fake_sample1)
                fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2
                fake_sample1 = torch.cat((real_img2, fake_sample1, rec_img2),axis=-1)
                torchvision.utils.save_image(fake_sample1, os.path.join(exp_path, 'sample1_discrete_epoch_{}.png'.format(epoch)), normalize=True)

                fake_sample2, _ = sampler.sample(model=gen_diffusive_2, batch_size=real_data1.shape[0], shape=real_data1.shape[1:], 
                                                 conditioning=real_data1, cond_for_CrossAttn=cond_data_1)
                fake_sample2 = AE.decode(fake_sample2)
                fake_sample2 = (torch.clamp(fake_sample2, -1, 1) + 1) / 2
                fake_sample2 = torch.cat((real_img1, fake_sample2, rec_img1),axis=-1)
                torchvision.utils.save_image(fake_sample2, os.path.join(exp_path, 'sample2_discrete_epoch_{}.png'.format(epoch)), normalize=True)

                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)                    
                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_1to2.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_1to2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_2to1.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_2to1_{}.pth'.format(epoch)))                
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)

            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            with torch.no_grad():
                val_psnr_values_1to2=[]
                val_psnr_values_2to1=[]
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                # 2to1
                print(epoch)
                real_data = x_val
                source_data = y_val
                source_img = source_data

                source_data = AE.get_latent(source_data)
                
                # LDM
                fake_sample1, _ = sampler.sample(model=gen_diffusive_1, batch_size=source_data.shape[0], shape=source_data.shape[1:],
                                                conditioning=source_data, cond_for_CrossAttn=DWT_func(source_img, 2))
            
                fake_sample1 = AE.decode(fake_sample1)
                fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2

                real_data = to_range_0_1(torch.clamp(real_data, -1, 1))

                ssim_score = ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                ms_ssim_score = ms_ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                mae_score = torch.abs(fake_sample1 - real_data).mean(dim=(1,2,3))

                step = 1
                for i in range(0, real_data.shape[0], step):
                    real = real_data[i:i+step].cpu().numpy()
                    fake = fake_sample1[i:i+step].cpu().numpy()
                    val_psnr_values_2to1.append(psnr(real,fake, data_range=1.0))
                print("2to1---SSIM:%f | MS SSIM:%f | PSNR:%f | MAE:%f" % 
                    (ssim_score.mean(), ms_ssim_score.mean(), np.nanmean(np.array(val_psnr_values_2to1)), mae_score.mean()))

                # 1to2
                real_data = y_val
                source_data = x_val
                source_img = source_data

                source_data = AE.get_latent(source_data)
                        
                # LDM
                fake_sample1, _ = sampler.sample(model=gen_diffusive_2, batch_size=source_data.shape[0], shape=source_data.shape[1:], 
                                                conditioning=source_data, cond_for_CrossAttn=DWT_func(source_img, 2))
                
                fake_sample1 = AE.decode(fake_sample1)
                fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2
                
                real_data = to_range_0_1(torch.clamp(real_data, -1, 1))

                ssim_score = ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                ms_ssim_score = ms_ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                mae_score = torch.abs(fake_sample1 - real_data).mean(dim=(1,2,3))

                step = 1
                for i in range(0, real_data.shape[0], step):
                    real = real_data[i:i+step].cpu().numpy()
                    fake = fake_sample1[i:i+step].cpu().numpy()
                    val_psnr_values_1to2.append(psnr(real,fake, data_range=1.0))
                print("1to2---SSIM:%f | MS SSIM:%f | PSNR:%f | MAE:%f" % 
                    (ssim_score.mean(), ms_ssim_score.mean(), np.nanmean(np.array(val_psnr_values_1to2)), mae_score.mean()))
            
            import gc
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()
            
def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--latent_size', type=int, default=32,
                            help='size of latent')# latent_size
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int,
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #generator and training
    parser.add_argument("--net_type", default="normal")# net type
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')
    parser.add_argument('--AE_ckpt_path', help='path to AE')# AE_ckpt_path
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5, help='weightening of l1 loss part of diffusion ans cycle models')
   
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--port_num', type=str, default='6021',
                        help='port selection for code')

   
    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train_syndiff, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        
        init_processes(0, size, train_syndiff, args)
