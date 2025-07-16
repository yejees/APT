import argparse
import os
import numpy as np
import torch as th

print(__file__)
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from pathlib import Path
from glob import glob
from tqdm import tqdm
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")
from open_clip import create_model_from_pretrained, get_tokenizer
from adapter.adapter import Adapter
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

def adapter_load_model(model, checkpoint_path):
    checkpoint = th.load(checkpoint_path, map_location=th.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs


def main():
    args = create_argparser().parse_args()
    args.save_dir = f'{args.save_dir}_{args.sample_method}'
    
    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    
    adapter = Adapter()
    adapter = adapter_load_model(adapter, './adapter/checkpoints/last.pth')
    adapter.to(dist_util.dev())
    adapter.eval()
    
    c_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    c_model = c_model.to(dist_util.dev())
    c_model.eval()
    
    out_path = Path(args.save_dir) / 'output'

    gt_path_all = Path(args.save_dir) / 'GT_all'
    gt_path_flair = Path(args.save_dir) / 'GT_flair'
    gt_path_t1 = Path(args.save_dir) / 'GT_t1'
    gt_path_t1ce = Path(args.save_dir) / 'GT_t1ce'
    gt_path_t2 = Path(args.save_dir) / 'GT_t2'

    out_path.mkdir(parents=True, exist_ok=True)
    gt_path_all.mkdir(parents=True, exist_ok=True)
    gt_path_flair.mkdir(parents=True, exist_ok=True)
    gt_path_t1.mkdir(parents=True, exist_ok=True)
    gt_path_t1ce.mkdir(parents=True, exist_ok=True)
    gt_path_t2.mkdir(parents=True, exist_ok=True)

    flair_path = []   
    flair_path.extend(glob(f'{args.data_path}/flair/*')) 
    flair_path.sort()


    for _, gt1_path in tqdm(enumerate(flair_path),total=len(flair_path)):
        subject_name = gt1_path.split('/')[-1]
        subject_num  = subject_name.split('_')[2]

        subject_name = subject_name.split('_')[1]
        gt1 = np.load(gt1_path)
        gt2_path = gt1_path.replace('flair','t1')
        gt2 = np.load(gt2_path)
        gt3_path = gt1_path.replace('flair','t1ce')
        gt3 = np.load(gt3_path)
        gt4_path = gt1_path.replace('flair','t2')
        gt4 = np.load(gt4_path)
        
        contrast1 = gt1[:,:,0]
        contrast1 = resize(contrast1,(args.img_size,args.img_size))
        contrast1 = contrast1[...,None] 
        c1_abs_t = abs(contrast1)
        c1_abs = c1_abs_t * 2 -1
        c1_abs_temp = c1_abs[None,...]
        
        contrast2 = gt2[:,:,0]
        contrast2 = resize(contrast2,(args.img_size,args.img_size))
        contrast2 = contrast2[...,None] 
        c2_abs_t = abs(contrast2)
        c2_abs = c2_abs_t * 2 -1
        c2_abs_temp = c2_abs[None,...]
        
        contrast3 = gt3[:,:,0]
        contrast3 = resize(contrast3,(args.img_size,args.img_size))
        contrast3 = contrast3[...,None] 
        c3_abs_t = abs(contrast3)
        c3_abs = c3_abs_t * 2 -1
        c3_abs_temp = c3_abs[None,...]
        
        contrast4 = gt4[:,:,0]
        contrast4 = resize(contrast4,(args.img_size,args.img_size))
        contrast4 = contrast4[...,None] 
        c4_abs_t = abs(contrast4)
        c4_abs = c4_abs_t * 2 -1
        c4_abs_temp = c4_abs[None,...]
        
        data_np = np.concatenate((c1_abs_temp,c2_abs_temp,c3_abs_temp,c4_abs_temp),axis=0)
        data = th.from_numpy(data_np)

        data = data.unsqueeze(0)
        data = data.squeeze(-1)
        data = data.to(dist_util.dev()).float()

        data_permute = data.permute(1,0,2,3)
        utils.save_image(data_permute, str(gt_path_all / f'{subject_name}-{subject_num}.png'), nrow = 4, normalize=True)

        utils.save_image(data[:,0:1], str(gt_path_flair / f'{subject_name}-{subject_num}.png'), nrow = 1, normalize=True)
        utils.save_image(data[:,1:2], str(gt_path_t1 / f'{subject_name}-{subject_num}.png'), nrow = 1, normalize=True)
        utils.save_image(data[:,2:3], str(gt_path_t1ce / f'{subject_name}-{subject_num}.png'), nrow = 1, normalize=True)
        utils.save_image(data[:,3:4], str(gt_path_t2 / f'{subject_name}-{subject_num}.png'), nrow = 1, normalize=True)
        
        contrast_num = args.contrast_num  
        contrast_num2 = args.contrast_num2
        contrast_num3 = args.contrast_num3
        
        c_name = 'flair'
        if contrast_num ==1:
            c_name = 't1'
        elif contrast_num ==2:
            c_name = 't1ce'
        elif contrast_num ==3:
            c_name = 't2'
        
        if contrast_num2:
            c_name2 = 'flair'
            if contrast_num2 ==1:
                c_name2 = 't1'
            elif contrast_num2 ==2:
                c_name2 = 't1ce'
            elif contrast_num2 ==3:
                c_name2 = 't2'
            out_path_c2 = str(out_path) + '/' + c_name2
            os.makedirs(out_path_c2,exist_ok=True)
        
        if contrast_num3:
            c_name3 = 'flair'
            if contrast_num3 ==1:
                c_name3 = 't1'
            elif contrast_num3 ==2:
                c_name3 = 't1ce'
            elif contrast_num3 ==3:
                c_name3 = 't2'
            out_path_c3 = str(out_path) + '/' + c_name3
            os.makedirs(out_path_c3,exist_ok=True)

        mask = th.ones_like(data).to(dist_util.dev())
        mask[:,contrast_num] = 0
        
        if contrast_num2:
            mask[:,contrast_num2] = 0
            
        out_path_c = str(out_path) + '/' + c_name
        os.makedirs(out_path_c,exist_ok=True)
        
        mask[:] = 1
        mask[:,contrast_num] = 0
        
        if contrast_num2:
            mask[:,contrast_num2] = 0
            
        if contrast_num3:
            mask[:,contrast_num3] = 0
        
        mask = mask.type(th.float32)
        
        sample = diffusion.p_sample_loop_grad(
            model,
            c_model,
            adapter,
            (args.batch_size, 4, args.image_size, args.image_size),
            data,
            mask,
            args.start_idx1,
            clip_denoised=args.clip_denoised, 
            progress=True,
        )
        
        sample = diffusion.p_sample_loop_last(
            sample,
            model,
            c_model,
            adapter,
            (args.batch_size, 4, args.image_size, args.image_size),
            data,
            mask,
            args.start_idx2,
            args.total_loop_num2,
            clip_denoised=args.clip_denoised, 
            progress=True,
        )
        
        gt_shift = 0.5*(data[:,contrast_num:contrast_num+1]+1)
        masking = (gt_shift > 0.01).float()
        
        sample_shift = (sample[:,contrast_num:contrast_num+1]+1)*0.5
        
        gt_shift = gt_shift * masking
        sample_shift = sample_shift * masking
        gt_shift /= th.max(gt_shift)
        sample_shift /= th.max(sample_shift)
        
        
        sample_temp = sample.permute(1,0,2,3)
        print('save path: ',str(f'{out_path_c}/{subject_name}-{subject_num}.png'))
        utils.save_image(sample_temp[contrast_num:contrast_num+1], str(f'{out_path_c}/{subject_name}-{subject_num}.png'), nrow = 1,normalize=True)

        if contrast_num2:
            print('save path: ',str(f'{out_path_c2}/{subject_name}-{subject_num}.png'))
            utils.save_image(sample_temp[contrast_num2:contrast_num2+1], str(f'{out_path_c2}/{subject_name}-{subject_num}.png'), nrow = 1,normalize=True)

        if contrast_num3:
            print('save path: ',str(f'{out_path_c3}/{subject_name}-{subject_num}.png'))
            utils.save_image(sample_temp[contrast_num3:contrast_num3+1], str(f'{out_path_c3}/{subject_name}-{subject_num}.png'), nrow = 1,normalize=True)
                

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=1,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        data_path="",
        save_latents=False,
        start_idx1=20,
        start_idx2=2,
        total_loop_num2=50,
        img_size=256,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default='MCG', help='One of [vanilla, MCG, repaint]')
    parser.add_argument('--mask_type', type=str, default='box', help='kind of box')
    parser.add_argument('--repeat_steps', type=int, default=10, help='For REPAINT, number of repeat steps')
    parser.add_argument('--contrast_num', type=int, default=0, help='target contrast')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory where the results will be saved')
    parser.add_argument('--contrast_num2', type=int, default=None, help='target contrast2')
    parser.add_argument('--contrast_num3', type=int, default=None, help='target contrast3')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()