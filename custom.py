import torch
import numpy as np
from glob import glob
import os
from torchvision import transforms

def min_max_norm(x):
    _min = x.min()
    _max = x.max()
    if _max - _min != 0:
        return (x - _min) / (_max - _min)
    else:
        return x

def imageshift(x):
    return (x * 2) - 1


class CustomDataset_brats(torch.utils.data.Dataset):
    def __init__(self,root,train=False):
        self.files = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256)),
            transforms.CenterCrop(256),
            imageshift,
        ])
        
        if train == True:
            path = os.path.join(root,'train/t2/*')
            file_list = glob(path)
            self.files += file_list
        
        else:
            path = os.path.join(root,'valid/t2/*')
            file_list = glob(path)
            self.files += file_list


    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
 	  

        file_flair = self.files[i].replace('t2','flair')
        flair_1 = np.load(file_flair)
      
        file_t1 = self.files[i].replace('t2','t1')
        t1_1 = np.load(file_t1)
	
        file_t1ce = self.files[i].replace('t2','t1ce')
        t1ce_1 = np.load(file_t1ce)
 	
        t2_1 = np.load(self.files[i])
        
        flair_1 = flair_1.astype(np.float32)
        t1_1 = t1_1.astype(np.float32)
        t1ce_1 = t1ce_1.astype(np.float32)
        t2_1 = t2_1.astype(np.float32)
        
        out = torch.cat((self.transform(flair_1), self.transform(t1_1), self.transform(t1ce_1), self.transform(t2_1)),dim=0)
        
        return out






