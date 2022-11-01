
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

parser = argparse.ArgumentParser(description='solve')
parser.add_argument('--cudaID', default=0, type=int,
                    help='cuda index')

args = parser.parse_args()
dev_ = 'cuda: ' + str(args.cudaID)
args.dev = torch.device( dev_ )


def mu_cov( h ):
    try:
        print( h.size(2), 'h has shape C x b x p' )
        bs = h.size(1)
        mu = h.mean( 1 ) ## shape C x p
        Dsquare = (torch.einsum('cbi,cbj->cbij', h, h)).mean( 1 ) ## shape C x p x p
        cov = ( Dsquare -  torch.einsum('ci,cj->cij', mu, mu) ) * bs / ( bs - 1. )
    except:
        print('h has shape b x p')
        bs = h.size(0)  ## shape b x p 
        mu = h.mean( 0 ) ## shape 1 x p
        Dsquare = (torch.einsum('bi,bj->bij', h, h)).mean( 0 ) ## shape  p x p
        cov = ( Dsquare -  torch.einsum('i,j->ij', mu, mu) ) * bs / ( bs - 1. )
        mu = mu.unsqueeze(0)
    return mu, cov, Dsquare, bs

def pca_( ckpt_list ):
    mu_cov_dict = {}
    for ckpt in ckpt_list:
        state = torch.load( "./runs/save_data/" + 'df_ckpt' + str(ckpt) + '.pth' ) 
        h_u, h_c = state['df'], state['df_c']
        h_u, h_c = h_u.to(args.dev), h_c.to(args.dev)
        print(h_u.shape, h_c.shape, 'shapes')
        h_ = torch.cat( (h_u, h_c.reshape( -1, h_u.size(1) )), 0  )
        (_, S, V )= torch.pca_lowrank(h_, q=100, center=False)
        pca_u = torch.matmul(h_u, V)
        pca_c = torch.einsum( 'cbi,ij->cbj', h_c, V )
        print( torch.norm(pca_c)/ torch.norm( h_c ), 'projection rate' )
        print( torch.norm(pca_u)/ torch.norm( h_u ),  'projection rate')
        mu, cov, D2, n_ = mu_cov( pca_u )  
        mu_c, cov_c, D2_c, n_c = mu_cov( pca_c )   
        mu_cov_dict[str(ckpt)] = ( mu, cov, D2, n_, mu_c, cov_c, D2_c, n_c )
    return mu_cov_dict

 
def Kl_d(mu0, cov0, mu1, cov1):
    ### KL( p0 || p1 )
    #cov1_inv, mu_ = torch.inverse( cov1 ), mu1 - mu0
    #l1 = torch.trace( torch.matmul( cov1_inv, cov0) ) 
    #l2 = torch.matmul( mu_, torch.matmul( cov1_inv, mu_.T ) )
    #l3 = torch.det( cov1).log()
    #l2_d = (( mu0 - mu1 )** 2).sum() + ( ( cov0- cov1 )**2 ).sum()
    c0, c1 = float( torch.trace(cov0).cpu() ), float( torch.trace(cov1).cpu() )
    l1 = (( mu0 - mu1 )** 2).sum()
    l2 = ( ( ( cov0 / c0- cov1 /c1 ).exp() + ( cov1 / c1 - cov0/c0 ).exp() ) *0.5 ).log().sum()
    return l1, l2 
  
def compute_loss(alpha, ckpt, mu_cov_dict ): 
    mu, cov, mu_c, cov_c = mu_cov_dict[str(ckpt)]
    mu_a = torch.matmul( alpha, mu_c)
    cov_a = torch.einsum( 'c,cij->ij', (alpha**2).view(-1), cov_c )
    l1, l2 = Kl_d(mu, cov, mu_a, cov_a)
    
    return l1, l2
 

ckpt_list = [ 1, 2, 5, 10, 25, 50, 75, 100, 150] 
mu_cov_dict = pca_(ckpt_list)
num_iterations = 100000
beta = torch.zeros( (1, 100), device =args.dev, requires_grad=True  )  ## initialize weight for each class
optimizer = torch.optim.SGD( [beta], lr=0.01, momentum=0.9 )
for itr in range(num_iterations):
    alpha = F.softmax( beta, dim =1 )   ## weight has shape 1 x C
    
    loss = 0. 
    for ckpt in ckpt_list:
        mu, cov, D2, n_, mu_c, cov_c, D2_c, n_c = mu_cov_dict[str(ckpt)]
        mu_a = torch.matmul( alpha, mu_c)
        cov_a_ = torch.einsum( 'c,cij->ij', (alpha**2).view(-1), cov_c )
        D2_a = torch.einsum( 'c,cij->ij', alpha.view(-1), D2_c )
        cov_a = D2_a - torch.matmul( mu_a.T, mu_a) + cov_a_ / n_c


        l1, l2 = Kl_d(mu, cov, mu_a, cov_a)
        loss += (l1 + l2) * ( 1./ len(ckpt_list ))
    entropy = (beta.exp().sum()).log() - (alpha * beta).sum()
    loss += 0.02 * entropy

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (itr+ 1)% 1000==0 or itr == 0:
        print( loss.item(), entropy.item(), itr+1 )
       

print(alpha)

torch.save( alpha.cpu(), './runs/save_data/alpha.pt' )






