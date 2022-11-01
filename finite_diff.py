
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import argparse

from utils.wideresnet import *
import time
import torch.nn.parallel
from tqdm import tqdm
#from lossutils import *

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cpt_f( net, input_ ):
    #### input has dim N x d, outputs dim NxC
    net.eval()
    with torch.no_grad():
        out = net( input_ )
    return torch.sqrt( F.softmax(out, dim =1) )

parser = argparse.ArgumentParser(description='WideResNet Compute Basis')
parser.add_argument('--cudaID', default=1, type=int,
                    help='cuda index')

def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    progress = tqdm(val_loader)
    for input, target in progress:
        target = target.to(args.dev, non_blocking=True)
        input = input.to(args.dev, non_blocking=True)

        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            #print(loss.item())

        acc1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))

        progress.set_description('Test Epoch: [{}/{}], Loss: {:.4f}, Acc@1: {:.3f}'.format(epoch, 0, losses.avg, top1.avg))
    
    return top1.avg


def main():
    #global args, best_acc1
    global args

    args = parser.parse_args()
    dev_ = 'cuda: ' + str(args.cudaID)
    args.dev = torch.device( dev_ )
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

    train_loader = DataLoader(datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True),
        batch_size=50000, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True),
        batch_size=1000, shuffle=True, num_workers=4, pin_memory=True)

    # model
    model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model = model.to(args.dev)
    
    ## cross entropy loss
    criterion = nn.CrossEntropyLoss().to(args.dev)
    
    ## load the learning rate schedult at each checkpoint
    lr_list = torch.load( './runs/save_data/lr_sche.pt' )
   
    for idx, ckpt in enumerate( [1, 2, 5, 10, 25, 50, 75, 100, 150, 200]):     ### list of the saved checkpoints
        
        # optimizer 
        lr_ = lr_list[idx][0]
        optimizer = torch.optim.SGD(model.parameters(), lr_)   ## modify the learning rate later
        
        # load checkpoint 
        checkpoint_ = 'runs/WideResNet-28-10/ckpt' + str(ckpt) + '.pth'
        checkpoint = torch.load( checkpoint_ )
        model.load_state_dict(checkpoint['state_dict'])
    
        ### load validation data
        val_iter = iter(val_loader)
        try:
            val_input, _ = next(val_iter)
        except:
            val_iter = iter(val_loader)
            val_input, _ = next(val_iter)
        val_input = val_input.to(args.dev, non_blocking=True)
           
        # load train data 
        data_iter = iter(train_loader)
        try:
            inputs, target = next(data_iter)
        except:
            data_iter = iter(train_loader)
            inputs, target = data_iter.next()
        target = target.to(args.dev, non_blocking=True)
        inputs = inputs.to(args.dev, non_blocking=True)

        ## seperate via classes
        id_list = []
        num_classes = 100 
        for c in range( num_classes ): 
            id_ = ( (target == c).nonzero(as_tuple=True)[0] ).cpu().tolist() # id_ has length 500 for each class c= 1,2,...,100
            id_list.append( id_ )
        f_start = cpt_f( model, val_input )

        state = {}
        #### compute universal gradient
        replica, bs = 100, 64
        save_ = 0. 
        for idx in range( replica ):
            left = int( bs * idx)
            right = left + bs 
            
            #model.train()
            model.eval()  ### the batch norm pamrams are sensitive. 
            #They heavily influence the results. We should freeze the batchnorm
            
            input_, target_ = inputs[left: right], target[left: right ] 
            out_ = model(input_) 
            loss = criterion( out_, target_ )

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            f_end = cpt_f( model, val_input )
            
            d_f = (f_end - f_start).flatten().unsqueeze(0).cpu()
            try:
                save_ = torch.cat( ( save_, d_f ), dim =0  )   ### In the end of subloop, save_ has dimension 25 x 500 x 100 
            except:
                save_ = d_f + 0.

            ## re-load model
            model.load_state_dict(checkpoint['state_dict'])

        state['df'] = save_ + 0. 
    
        ###### compute per-category gradeint
        bs = 20
        #replica = 500 // bs
        replica = 500 // bs
        t0 = time.time()
        save_f = 0. 
        for c in range( 100 ):
            save_ = 0. 
            for idx in range( replica ):
                left = int( bs * idx)
                right = left + bs 
                id_ = id_list[c][left: right]        
            
                #model.train()
                model.eval()  ### the batch norm pamrams are sensitive. 
                #They heavily influence the results. We should freeze the batchnorm
                
                input_, target_ = inputs[id_], target[id_] 
                out_ = model(input_) 
                loss = criterion( out_, target_ )

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                f_end = cpt_f( model, val_input )
                
                d_f = (f_end - f_start).flatten().unsqueeze(0).cpu()
                try:
                    save_ = torch.cat( ( save_, d_f ), dim =0  )   ### In the end of subloop, save_ has dimension 25 x 500 x 100 
                except:
                    save_ = d_f + 0.

                ## re-load model
                model.load_state_dict(checkpoint['state_dict'])

            save_ = save_.unsqueeze(0)
            try:
                save_f = torch.cat( ( save_f, save_ ), dim =0  )
            except:
                save_f = save_ + 0. 
            print( time.time() - t0, c, loss.item(), out_.shape )
       
        state['df_c'] = save_f + 0. 

        directory = "runs/save_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + 'df_ckpt' + str(ckpt) + '.pth'
        print(save_f.shape, 'save shape')
        torch.save(state, filename)
        
    acc1 = validate(val_loader, model, criterion, 0)

if __name__ == '__main__':
    main()
