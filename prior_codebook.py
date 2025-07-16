import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt

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
from utils import clear_color, normalize_np, clear, prepare_im, psnr, SSIM, rmse, normalize
import math
from glob import glob
from tqdm import tqdm
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")
from open_clip import create_model_from_pretrained, get_tokenizer
from adapter.adapter import Adapter
import random
import torch.nn.functional as F
from bezier_curve import nonlinear_transformation

seed = 1
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

def get_second_most_common(arr):
    # Get counts of each unique value
    counts = np.bincount(arr)
    # Get indices that would sort counts in descending order
    sorted_indices = np.argsort(-counts)
    # Return the second element (index 1) if it exists, otherwise return the first element
    return sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]

def get_third_most_common(arr):
    # Get counts of each unique value
    counts = np.bincount(arr)
    # Get indices that would sort counts in descending order
    sorted_indices = np.argsort(-counts)
    # Return the third element (index 2) if it exists, otherwise return the last available element
    return sorted_indices[2] if len(sorted_indices) > 2 else sorted_indices[-1]

def get_fourth_most_common(arr):
    # Get counts of each unique value
    counts = np.bincount(arr)
    # Get indices that would sort counts in descending order
    sorted_indices = np.argsort(-counts)
    # Return the third element (index 2) if it exists, otherwise return the last available element
    return sorted_indices[3] if len(sorted_indices) > 3 else sorted_indices[-2]

def get_fifth_most_common(arr):
    # Get counts of each unique value
    counts = np.bincount(arr)
    # Get indices that would sort counts in descending order
    sorted_indices = np.argsort(-counts)
    # Return the third element (index 2) if it exists, otherwise return the last available element
    return sorted_indices[4] if len(sorted_indices) > 4 else sorted_indices[-3]


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
    c_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    c_model = c_model.to(dist_util.dev())
    c_model.eval()

    all_flair_path = []   
    flair_path = data_path + '/train/flair/'
    all_flair_path.extend(glob(flair_path + '*'))

    print(len(flair_path))
    
    top_1_ind_all = th.empty(0,8).to(dist_util.dev()) 
    top_1_mean_all = th.empty(0,8).to(dist_util.dev())
    for i, gt1_path in tqdm(enumerate(flair_path),total=len(flair_path)):

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
        contrast1 = resize(contrast1,(256,256))
        contrast1 = contrast1[...,None] 
        c1_abs_t = abs(contrast1)
        c1_abs = c1_abs_t * 2 -1
        c1_abs_temp = c1_abs[None,...]
        
        contrast2 = gt2[:,:,0]
        contrast2 = resize(contrast2,(256,256))
        contrast2 = contrast2[...,None] 
        c2_abs_t = abs(contrast2)
        c2_abs = c2_abs_t * 2 -1
        c2_abs_temp = c2_abs[None,...]
        
        contrast3 = gt3[:,:,0]
        contrast3 = resize(contrast3,(256,256))
        contrast3 = contrast3[...,None] 
        c3_abs_t = abs(contrast3)
        c3_abs = c3_abs_t * 2 -1
        c3_abs_temp = c3_abs[None,...]
        
        contrast4 = gt4[:,:,0]
        contrast4 = resize(contrast4,(256,256))
        contrast4 = contrast4[...,None] 
        c4_abs_t = abs(contrast4)
        c4_abs = c4_abs_t * 2 -1
        c4_abs_temp = c4_abs[None,...]
        
        data_np = np.concatenate((c1_abs_temp,c2_abs_temp,c3_abs_temp,c4_abs_temp),axis=0)
        data = th.from_numpy(data_np)

        data = data.unsqueeze(0)
        data = data.squeeze(-1)
        data = data.to(dist_util.dev()).float()
                    
        batch, num, h, w = data.shape
        mul_cond = data.repeat(batch,1,1,1)
        cond_flatten  = mul_cond.view(batch*num,h,w)
        
        cond_flatten = cond_flatten.cpu().numpy()
        cond_flatten = cond_flatten*0.5 + 0.5
        
        
        cond_flatten = (cond_flatten-np.min(cond_flatten)) / (np.max(cond_flatten)-np.min(cond_flatten))

        
        all_cond = []
        for j in range(num):
            image1 = cond_flatten[j,:,:]
            
            image1, image2 = nonlinear_transformation(image1)    
            image1 = image1[None,None,...]
            image2 = image2[None,None,...]

            image2 = (image2 - np.min(image2))
            
            masking = (image1 > 0.04).astype(np.float32)
            image2 = image2 * masking
            all_cond.append(image1)
            all_cond.append(image2)
        
        all_cond = np.array(all_cond)
        all_cond = np.concatenate(all_cond, axis=0)
        
        all_cond = th.from_numpy(all_cond).to(dist_util.dev()).float()
        cond_flatten_r = all_cond.repeat(1,3,1,1)
        upsampled_size = (224, 224)

        with th.no_grad():
            cond_flatten_r_resize = F.interpolate(cond_flatten_r, size=upsampled_size, mode='bilinear', align_corners=False)
            cond_features = c_model.encode_image(cond_flatten_r_resize*2-1)
            cond_features /= cond_features.norm(dim=-1, keepdim=True)
            
        attn_map = cond_features @ cond_features.T
        for k in range(0, 8, 2):
            attn_map[k:k+2, k:k+2] = -100
            
        attn = F.softmax(attn_map, dim=1) 
        
        plt.imshow(attn.cpu().numpy(), cmap='jet', vmin=0, vmax=1.1*np.max(attn.cpu().numpy()))
        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.savefig(f'{save_path_attn}/{subject_name}.png',bbox_inches='tight',pad_inches=0)
        plt.close()

        _, top_1_ind = th.max(attn, dim=1)
        
        top_1_ind = top_1_ind.int()
        top_1_ind = top_1_ind.unsqueeze(0)   
        top_1_ind_all = th.cat((top_1_ind_all,top_1_ind),dim=0)
        
        all_mean = th.mean(cond_flatten_r[:,0,:,:],dim=(-2,-1))
        all_mean = all_mean.unsqueeze(0)
        top_1_mean_all = th.cat((top_1_mean_all,all_mean),dim=0)
    
    top_1_mean_all = th.mean(top_1_mean_all,dim=0)
    
    ratio_matrix = th.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            ratio_matrix[i,j] = top_1_mean_all[i] / top_1_mean_all[j]
            
    print("\nRatio matrix:")
    print(ratio_matrix)
    
    top_1_ind_all_np = top_1_ind_all.cpu().numpy()
    top_1_ind_all_np = top_1_ind_all_np.astype(np.int64)
    
    # Find the most common value for each column
    most_common_per_column = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=top_1_ind_all_np)
    
    print("Most common values for each column:")
    print(most_common_per_column)
    
    second_most_common_per_column = np.apply_along_axis(get_second_most_common, axis=0, arr=top_1_ind_all_np)
    print("Second most common values for each column:")
    print(second_most_common_per_column)
    
    third_most_common_per_column = np.apply_along_axis(get_third_most_common, axis=0, arr=top_1_ind_all_np)
    print("Third most common values for each column:")
    print(third_most_common_per_column)
    
    fourth_most_common_per_column = np.apply_along_axis(get_fourth_most_common, axis=0, arr=top_1_ind_all_np)
    print("Fourth most common values for each column:")
    print(fourth_most_common_per_column)
    
    fifth_most_common_per_column = np.apply_along_axis(get_fifth_most_common, axis=0, arr=top_1_ind_all_np)
    print("Fifth most common values for each column:")
    print(fifth_most_common_per_column)
    
    # Optional: Calculate and print the frequency of the most common value in each column
    column_frequencies = np.apply_along_axis(lambda x: np.bincount(x).max(), axis=0, arr=top_1_ind_all_np)
    print("\nFrequency of most common values:")
    print(column_frequencies)
    
    # Optional: Calculate and print the percentage of the most common value in each column
    column_percentages = column_frequencies / len(top_1_ind_all_np) * 100
    print("\nPercentage of most common values:")
    print(column_percentages)
    

def create_argparser():
    defaults = dict(
        data_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory where the results will be saved')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()