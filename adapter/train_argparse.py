import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CLIP fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/home/compu/yj/data/BraTS2021_2d', help='data directory')  
    parser.add_argument('--save_interval', type=int, default=50, help='save interval')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save directory')
    parser.add_argument('--log_dir', type=str, default='./checkpoints/logs', help='log directory')
    
    args = parser.parse_args()
    return args

# print(args)