import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from tqdm import trange
import time
from scipy.special import comb, perm
import numpy as np

args = {
    'dev': torch.device('cuda: 1' ),
    'num_dtheta': 1000,
    'm': 10,
    'n': 11,
    'max_iter': 5e4,
    'lr': 5. 
}


# In[2]:


num_dtheta, m, n = args['num_dtheta'], args['m'], args['n']
tail = 0.5 / num_dtheta
theta = torch.linspace(tail, 1. - tail, num_dtheta).reshape( num_dtheta, 1).to(args['dev'])
prior = (torch.ones_like(theta ) / (num_dtheta )).to(args['dev'])


# In[3]:


def gadient( m, args, theta, prior ):
    #list of number of heads
    heads = torch.cat( [torch.tensor([ k ]) for k in range(m+1)], 0 ).unsqueeze(1).to(args['dev'])
    #combination numbers
    cb = torch.cat( [ torch.tensor([comb(m, k )]) for k in range(m+1)], 0 ).unsqueeze(1).to(args['dev'])
    #p(x| theta)
    p_x_th = (torch.pow( theta,  heads.T)) * (torch.pow( 1.0 -  theta,  (m - heads).T))
    # log p(x| theta)
    log_p_x_th = (theta.log()) * heads.T + (( 1. - theta ).log()) * (( m - heads ).T )
    #p(x)
    p_x = ( prior *  p_x_th).sum(0).unsqueeze(0)
    # f_kl function at theta
    f_kl_th = m * ( theta * (theta.log() )+ (1. - theta) * ( (1. - theta).log()) ) - ((cb.T * p_x_th) * (   p_x.log() )).sum(1).unsqueeze(1)
    # d p(x|th) / d theta
    grad_p = (torch.pow( theta,  (heads - 1).T)) * (torch.pow( 1.0 -  theta,  (m - 1 - heads).T)) * ( heads.T- m * theta )
    #mutual information I(x:theta)
    MI = (prior * f_kl_th).sum()
    #d theta
    d_th1 = (prior ** 2) * ( ((cb.T * p_x_th) * (grad_p/ p_x)).sum(1).unsqueeze(1))
    d_th = prior * (((cb.T * grad_p) * ( log_p_x_th - p_x.log() )).sum(1).unsqueeze(1)) - d_th1
    
    return d_th, f_kl_th, MI 


# In[4]:


for iteration in range(20000):
    
    d_th1, f_kl1, MI1 = gadient( m, args, theta, prior )
    d_th2, f_kl2, MI2 = gadient( n, args, theta, prior )
    
    #divergences
    d_th = d_th2 - d_th1
    f_kl = f_kl2 - f_kl1
    MI = MI2 - MI1
    #uodate theta
    theta += args['lr'] * d_th / torch.norm(d_th)
    #update prior
    prior1 = prior * (f_kl.exp())
    prior = prior1 / prior1.sum()
    
    theta += (1e-4 - theta).relu()
    theta -= (theta - 1. + 1e-4).relu() 
    prior = prior.relu()
    prior = prior / prior.sum()
    
    if (iteration +1)%5000 ==0 or iteration == 0:
        print( iteration+1, MI.data.cpu().numpy() )


# In[7]:


count = torch.round( 100000 * prior )
samples = torch.cat( [ theta[k].repeat( int(count[k].data) ) for k in range(num_dtheta)], 0 )
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')
d = {'activations': samples.view(-1).cpu().numpy()}
plt.figure(figsize=(6, 4))
plt.hist(d['activations'], color = 'green', edgecolor = 'black',
         bins = int(500))
plt.xlabel('theta')
plt.tight_layout()


