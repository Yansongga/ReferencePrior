import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from utils.wideresnet import WideResNet
from torch.utils.data import DataLoader
from tqdm import tqdm

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='WideResNet Pre-Training')
# training specs
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')


def main():
    global args
    args = parser.parse_args()
    #if args.tensorboard: configure("runs/%s"%(args.name))

    # data loading
    train_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])

    train_loader = DataLoader(datasets.__dict__[args.dataset.upper()](root='./data', train=True, transform=train_transform, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(datasets.__dict__[args.dataset.upper()](root='./data', train=False, transform=test_transform, download=True),
        batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

 

    cudnn.benchmark = True
    # training specs
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    
    lr_list = []
    torch.save( lr_list, './runs/save_data/lr_sche.pt' )
    for epoch in range(args.start_epoch, args.epochs):
        #progress = tqdm(train_loader)
        if (epoch+1) in [1, 2, 5, 10, 25, 50, 75, 100, 150, 200]:
            lr = scheduler.get_last_lr()
            lr_list.append( lr )
            print(epoch+1, lr, 'lr')
        #for input, target in progress:
        for idx in range(len( train_loader )):
            scheduler.step()
    
    torch.save( lr_list, './runs/save_data/lr_sche.pt' )
  


if __name__ == '__main__':
    main()
