
import argparse
import torch
import numpy as np

import os
import torchvision
from backbones.autoencoder import AutoencoderKL
from backbones.unet import UNetModel
from backbones.ddim import DDIMSampler
from dataset import CreateDatasetSynthesis
import glob
from pytorch_msssim import ssim, ms_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D

import random
np.random.seed(11)
random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

import json

dwt = DWT_2D("haar")
iwt = IDWT_2D("haar")

def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch, device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
   
    for key in list(ckpt.keys()):
        if 'module' in key:
            ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

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
def sample_and_test(args):
    torch.manual_seed(42)
    # device = 'cuda:0'
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen=args.which_epoch
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    #loading dataset
    phase=args.phase
    dataset=CreateDatasetSynthesis(phase, args)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4)

    G_NET_ZOO = {"unet": UNetModel}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))
    #Initializing and loading network
    args.image_size = args.latent_size
    concat_channels = args.num_channels
    tome_info = None
    gen_diffusive_1 = gen_net(image_size=args.latent_size,in_channels=concat_channels,out_channels=concat_channels,model_channels=192,
                                attention_resolutions=[1,2,4,8],num_res_blocks=2,channel_mult=[1,2,2,4,4],
                                num_heads=8,use_scale_shift_norm=True,resblock_updown=True,
                                use_spatial_transformer=False, use_spatial_transformer_bottlenet=False,
                                use_VRKWV=False, RKWV_resolutions=[],
                                context_dim=None,context_size=None,tome_info=tome_info)
    gen_diffusive_2 = gen_net(image_size=args.latent_size,in_channels=concat_channels,out_channels=concat_channels,model_channels=192,
                                attention_resolutions=[1,2,4,8],num_res_blocks=2,channel_mult=[1,2,2,4,4],
                                num_heads=8,use_scale_shift_norm=True,resblock_updown=True,
                                use_spatial_transformer=False, use_spatial_transformer_bottlenet=False,
                                use_VRKWV=False, RKWV_resolutions=[],
                                context_dim=None,context_size=None,tome_info=tome_info)
    args.num_channels = args.num_channels // 2

    # load AE
    AE = AutoencoderKL(embed_dim=4,double_z=True,z_channels=4,in_channels=1,out_ch=1)
    AE.init_from_ckpt(args.AE_ckpt_path, ['loss'])
    AE = AE.eval().to(device)
    for param in AE.parameters():
        param.requires_grad = False

    sampler = DDIMSampler(device=device, ddim_num_steps=args.ddim_num_steps, ddim_eta=1., verbose=True, beta_schedule="linear", timesteps=args.num_timesteps)
    
    lpips_model = lpips.LPIPS(net="alex").to(device)

    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir,exp)

    checkpoint_file = exp_path + "/{}_{}.pth"
    if len(epoch_chosen) == 1 and epoch_chosen[0] == -1:
        epoch_list = glob.glob(os.path.join(exp_path, 'gen_diffusive_1_*.pth'))
        # print(os.path.join(exp_path, 'gen_diffusive_1_*.pth'))
        epoch_list = [i.split('_')[-1].split('.')[0] for i in epoch_list]
        epoch_list.sort(key=lambda x:int(x), reverse=False)
    else:
        epoch_list = epoch_chosen
         
    if phase == 'test':
        save_img_dir = exp_path + "/generated_samples/epoch_{}_step_{}/imgs/".format(epoch_chosen[0], args.ddim_num_steps)
        save_txt_dir = exp_path + "/generated_samples/epoch_{}_step_{}".format(epoch_chosen[0], args.ddim_num_steps)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        if not os.path.exists(save_txt_dir):
            os.makedirs(save_txt_dir)
    
    if phase == 'val':
        names, _, x_val , y_val = next(iter(data_loader))
    
    with torch.no_grad():
        for epoch in epoch_list:
            ssim_val_1to2, ms_ssim_val_1to2, lpips_val_1to2= [], [], []
            val_psnr_values_1to2=[]
            
            ssim_val_2to1, ms_ssim_val_2to1, lpips_val_2to1= [], [], []
            val_psnr_values_2to1=[]
            
            if phase == 'test':
                if args.b2a:
                    metadata = []
                    gen_diffusive_1 = gen_diffusive_1.to(device)
                    load_checkpoint(checkpoint_file, gen_diffusive_1,'gen_diffusive_1',epoch=str(epoch), device = device)
                    for iteration, (names, _, x_val , y_val) in enumerate(data_loader): 
                
                        real_data = x_val.to(device, non_blocking=True)
                        source_data = y_val.to(device, non_blocking=True)
                        source_img = source_data

                        source_data = AE.get_latent(source_data)
                        
                        #diffusion steps
                        fake_sample1, _ = sampler.sample(model=gen_diffusive_1, batch_size=source_data.shape[0], shape=source_data.shape[1:], 
                        conditioning=source_data, cond_for_CrossAttn=DWT_func(source_img, 2))  
                        
                        fake_sample1 = AE.decode(fake_sample1)
                        fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2

                        real_data = to_range_0_1(torch.clamp(real_data, -1, 1))

                        ssim_score = ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                        ssim_val_2to1.append(ssim_score)
                        print(ssim_score.mean())

                        ms_ssim_score = ms_ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                        ms_ssim_val_2to1.append(ms_ssim_score)
                        
                        lpips_score = lpips_model(fake_sample1, real_data)
                        lpips_val_2to1.append(lpips_score)

                        step = 1
                        for i in range(0, real_data.shape[0], step):
                            real = real_data[i:i+step].cpu().numpy()
                            fake = fake_sample1[i:i+step].cpu().numpy()
                            p = psnr(real, fake, data_range=1.0)
                            val_psnr_values_2to1.append(p)
                            if args.save_img:
                                torchvision.utils.save_image(fake_sample1[i], os.path.join(save_img_dir, '2to1-' + names[i] + '.jpg'),normalize=False)
                                with open(os.path.join(save_txt_dir,'2to1-metric.txt'),"a") as f:
                                    f.write("2to1:{}-SSIM:{:.4f}-MS-SSIM:{:.6f}-PSNR:{:.4f}-LPIPS:{:.6f}\n".format(names[i], ssim_score[i].item(), ms_ssim_score[i].item(), p, lpips_score[i].item()))
                                metadata.append(
                                    {
                                        "name": names[i],
                                        "SSIM": ssim_score[i].item(),
                                        "PSNR": p,
                                        "LPIPS": lpips_score[i].item()
                                    }
                                )

                    ssim_val_2to1 = torch.cat(ssim_val_2to1, 0)
                    ms_ssim_val_2to1 = torch.cat(ms_ssim_val_2to1, 0)
                    lpips_val_2to1 = torch.cat(lpips_val_2to1, 0)
                    print(epoch)
                    print("2to1---SSIM:%f | MS SSIM:%f | PSNR:%f | LPIPS:%f" % (ssim_val_2to1.mean(), ms_ssim_val_2to1.mean(), np.nanmean(np.array(val_psnr_values_2to1)), lpips_val_2to1.mean()))
                    with open(os.path.join(save_txt_dir,'2to1-metric.txt'),"a") as f:
                        f.write("2to1-SSIM:{:.4f}-MS-SSIM:{:.6f}-PSNR:{:.4f}-LPIPS:{:.6f}\n".format(ssim_val_2to1.mean(), ms_ssim_val_2to1.mean(), np.nanmean(np.array(val_psnr_values_2to1)), lpips_val_2to1.mean()))
                    metadata.append(
                        {
                            "MS-SSIM": ms_ssim_val_2to1.mean().item(),
                            "SSIM": ssim_val_2to1.mean().item(),
                            "PSNR": np.nanmean(np.array(val_psnr_values_2to1)),
                            "LPIPS": lpips_val_2to1.mean().item()
                        }
                    )
                    with open(os.path.join(save_txt_dir,'2to1-metric.json'), "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=4) 


                if args.a2b:
                    metadata = []
                    gen_diffusive_2 = gen_diffusive_2.to(device)
                    load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive_2',epoch=str(epoch), device = device)
                    for iteration, (names, _, y_val , x_val) in enumerate(data_loader): 
                
                        real_data = x_val.to(device, non_blocking=True)
                        source_data = y_val.to(device, non_blocking=True)
                        source_img = source_data

                        source_data = AE.get_latent(source_data)
                        
                        #diffusion steps
                        fake_sample1, _ = sampler.sample(model=gen_diffusive_2, batch_size=source_data.shape[0], shape=source_data.shape[1:], 
                        conditioning=source_data, cond_for_CrossAttn=DWT_func(source_img, 2))   
                        
                        fake_sample1 = AE.decode(fake_sample1)
                        fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2

                        real_data = to_range_0_1(torch.clamp(real_data, -1, 1))

                        ssim_score = ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                        ssim_val_1to2.append(ssim_score)
                        print(ssim_score.mean())

                        ms_ssim_score = ms_ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                        ms_ssim_val_1to2.append(ms_ssim_score)
                        
                        lpips_score = lpips_model(fake_sample1, real_data)
                        lpips_val_1to2.append(lpips_score)

                        step = 1
                        for i in range(0, real_data.shape[0], step):
                            real = real_data[i:i+step].cpu().numpy()
                            fake = fake_sample1[i:i+step].cpu().numpy()
                            p = psnr(real, fake, data_range=1.0)
                            val_psnr_values_1to2.append(p)
                            if args.save_img:
                                torchvision.utils.save_image(fake_sample1[i], os.path.join(save_img_dir, '1to2-' + names[i] + '.jpg'),normalize=False)
                                with open(os.path.join(save_txt_dir,'1to2-metric.txt'),"a") as f:
                                    f.write("1to2:{}-SSIM:{:.4f}-MS-SSIM:{:.6f}-PSNR:{:.4f}-LPIPS:{:.6f}\n".format(names[i], ssim_score[i].item(), ms_ssim_score[i].item(), p, lpips_score[i].item()))
                                metadata.append(
                                    {
                                        "name": names[i],
                                        "SSIM": ssim_score[i].item(),
                                        "PSNR": p,
                                        "LPIPS": lpips_score[i].item()
                                    }
                                )

                    ssim_val_1to2 = torch.cat(ssim_val_1to2, 0)
                    ms_ssim_val_1to2 = torch.cat(ms_ssim_val_1to2, 0)
                    lpips_val_1to2 = torch.cat(lpips_val_1to2, 0)
                    print(epoch)
                    print("1to2---SSIM:%f | MS SSIM:%f | PSNR:%f | LPIPS:%f" % (ssim_val_1to2.mean(), ms_ssim_val_1to2.mean(), np.nanmean(np.array(val_psnr_values_1to2)), lpips_val_1to2.mean()))
                    with open(os.path.join(save_txt_dir,'1to2-metric.txt'),"a") as f:
                        f.write("1to2-SSIM:{:.4f}-MS-SSIM:{:.6f}-PSNR:{:.4f}-LPIPS:{:.6f}\n".format(ssim_val_1to2.mean(), ms_ssim_val_1to2.mean(), np.nanmean(np.array(val_psnr_values_1to2)), lpips_val_1to2.mean()))
                    metadata.append(
                        {
                            "MS-SSIM": ms_ssim_val_1to2.mean().item(),
                            "SSIM": ssim_val_1to2.mean().item(),
                            "PSNR": np.nanmean(np.array(val_psnr_values_1to2)),
                            "LPIPS": lpips_val_1to2.mean().item()
                        }
                    )
                    with open(os.path.join(save_txt_dir,'1to2-metric.json'), "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=4) 
            else:
                gen_diffusive_1 = gen_diffusive_1.to(device)
                load_checkpoint(checkpoint_file, gen_diffusive_1,'gen_diffusive_1',epoch=str(epoch), device = device)
                gen_diffusive_2 = gen_diffusive_2.to(device)
                load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive_2',epoch=str(epoch), device = device)
                # 2to1
                print(epoch)
                real_data = x_val.to(device, non_blocking=True)
                source_data = y_val.to(device, non_blocking=True)
                source_img = source_data

                source_data = AE.get_latent(source_data)
                
                #diffusion steps
                fake_sample1, _ = sampler.sample(model=gen_diffusive_1, batch_size=source_data.shape[0], shape=source_data.shape[1:], 
                conditioning=source_data, cond_for_CrossAttn=DWT_func(source_img, 2))   
                
                fake_sample1 = AE.decode(fake_sample1)
                fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2

                real_data = to_range_0_1(torch.clamp(real_data, -1, 1))

                ssim_score = ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                ms_ssim_score = ms_ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                lpips_score = lpips_model(fake_sample1, real_data)

                step = 1
                for i in range(0, real_data.shape[0], step):
                    real = real_data[i:i+step].cpu().numpy()
                    fake = fake_sample1[i:i+step].cpu().numpy()
                    val_psnr_values_2to1.append(psnr(real,fake, data_range=1.0))
                print("2to1---SSIM:%f | MS SSIM:%f | PSNR:%f | LPIPS:%f" % (ssim_score.mean(), ms_ssim_score.mean(), np.nanmean(np.array(val_psnr_values_2to1)), lpips_score.mean()))

                # 1to2
                real_data = y_val.to(device, non_blocking=True)
                source_data = x_val.to(device, non_blocking=True)
                source_img = source_data

                source_data = AE.get_latent(source_data)
                        
                #diffusion steps
                fake_sample1, _ = sampler.sample(model=gen_diffusive_2, batch_size=source_data.shape[0], shape=source_data.shape[1:], 
                conditioning=source_data, cond_for_CrossAttn=DWT_func(source_img, 2))   
                
                fake_sample1 = AE.decode(fake_sample1)
                fake_sample1 = (torch.clamp(fake_sample1, -1, 1) + 1) / 2

                real_data = to_range_0_1(torch.clamp(real_data, -1, 1))

                ssim_score = ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                ms_ssim_score = ms_ssim(fake_sample1, real_data, data_range=1.0, size_average=False)
                lpips_score = lpips_model(fake_sample1, real_data)

                step = 1
                for i in range(0, real_data.shape[0], step):
                    real = real_data[i:i+step].cpu().numpy()
                    fake = fake_sample1[i:i+step].cpu().numpy()
                    val_psnr_values_1to2.append(psnr(real,fake, data_range=1.0))
                print("1to2---SSIM:%f | MS SSIM:%f | PSNR:%f | LPIPS:%f" % (ssim_score.mean(), ms_ssim_score.mean(), np.nanmean(np.array(val_psnr_values_1to2)), lpips_score.mean()))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
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
    parser.add_argument('--attn_resolutions', default=(16,),
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
    parser.add_argument('--AE_ckpt_path', help='path to AE')# AE_ckpt_path

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)

    parser.add_argument('--b2a', action='store_true',default=False) # 2 to 1
    parser.add_argument('--a2b', action='store_true',default=False) # 1 to 2
    parser.add_argument('--save_img', action='store_true',default=False) # if save test iamges
    
    #generator and training
    parser.add_argument("--net_type", default="normal")# net type
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')
    parser.add_argument('--phase', type=str, default='train', help='train, val or test') # train, val or test

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--latent_size', type=int, default=32,
                            help='size of latent')# latent_size
    parser.add_argument('--concat_channels', type=int, default=8)

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--ddim_num_steps', type=int, default=200)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')
    
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--which_epoch', nargs='+', type=int)
    parser.add_argument('--gpu_chose', type=int, default=0)


    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast')   
    args = parser.parse_args()
    
    sample_and_test(args)
    
