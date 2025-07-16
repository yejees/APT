import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data.distributed import DistributedSampler
from custom import CustomDataset_brats
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
import dist_util
from train_argparse import get_args
from adapter import Adapter
from tqdm import tqdm
import random
from pathlib import Path
from gather import all_gather
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import csv

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, epoch, loss, args, is_latest=False):
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if is_latest:
        save_path = save_dir / 'latest_checkpoint.pth'
    else:
        save_path = save_dir / f'epoch_{epoch}.pth'
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
    
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
        

def create_datasets(args):
    train_data = CustomDataset_brats(
        root=args.data_dir,
        train=True
    )
    # val_data = CustomDataset_brats(
    #     root=args.data_dir,
    #     train=False
    # )
    return train_data
  
def create_data_loaders_train(args):
    train_data = create_datasets(args)
    train_sampler = DistributedSampler(train_data,shuffle=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
        
    )
    return train_loader

def main(args,local_rank=0):
  
  if local_rank == 0:
    writer = SummaryWriter(log_dir=args.log_dir)
    log_file = os.path.join(args.log_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer2 = csv.writer(f)
        writer2.writerow(['Epoch', 'Loss'])

  train_dataloader = create_data_loaders_train(args)

  c_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
  c_model = c_model.to(dist_util.dev())
  c_model = DDP(c_model, device_ids=[local_rank])
  c_model.eval()

  model = Adapter().to(dist_util.dev())
  model = DDP(model, device_ids=[local_rank])

  loss_img = nn.CrossEntropyLoss()
  loss_img2 = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

  best_loss = float('inf')
  for epoch in range(args.epochs):
    epoch_loss = 0
    for batch in tqdm(train_dataloader) :
      optimizer.zero_grad()

      images = batch
      images = images.to(dist_util.dev())
      ######feature extraction from biomedclip######
      batch, ch, he, wi = images.shape
      images_flatten = images.view(batch*ch,1,he,wi)
      images_flatten_r = images_flatten.repeat(1,3,1,1)
      upsampled_size = (224, 224)
      images_itp = F.interpolate(images_flatten_r, size=upsampled_size, mode='bilinear', align_corners=False)
      
      with torch.no_grad():
          image_features = c_model.module.encode_image(images_itp)
      img_features = image_features.view(batch,ch,-1) 
  
      img_features = img_features.to(dist_util.dev())  
      rand_num = random.randint(0,ch-1)
      num_list = torch.arange(ch)
      filtered_num_list = num_list[num_list != rand_num]
      img_features_rand = img_features[:,filtered_num_list,:]

      image_features1, image_features2, logit_scale = model(img_features, img_features_rand)
      all_image_features1 = all_gather(image_features1)
      all_image_features2 = all_gather(image_features2)
      
      logits_per_image = logit_scale * all_image_features1 @ all_image_features2.t()
      logits_per_image2 = logits_per_image.t()

      ground_truth = torch.arange(len(all_image_features1),dtype=torch.long,device=dist_util.dev())

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_img2(logits_per_image2,ground_truth))/2
      
      total_loss.backward()
      optimizer.step()
    
      epoch_loss += total_loss.item()
    avg_epoch_loss = epoch_loss / len(train_dataloader)

    if local_rank == 0:

        print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss}")
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        with open(log_file, 'a', newline='') as f:
          writer2 = csv.writer(f)
          writer2.writerow([epoch, avg_epoch_loss])

        if (epoch + 1) % args.save_interval == 0:
          save_checkpoint(model, optimizer, epoch, avg_epoch_loss, args)
        
          save_checkpoint(model, optimizer, epoch, avg_epoch_loss, args, is_latest=True)
        
          if avg_epoch_loss < best_loss:
              best_loss = avg_epoch_loss
              save_checkpoint(model, optimizer, epoch, avg_epoch_loss, args, is_latest=False)
              print(f"New best model saved at epoch {epoch}")



if __name__ == "__main__":
  local_rank = dist_util.setup_dist()
  args = get_args()


  main(args,local_rank=local_rank)