import torch
import torch.nn.functional as F 

import pdb
def loss_function(recon_x,x,mu,sig_var):
	reconstruction_loss = F.mse_loss(recon_x,x.view(-1,784),size_average =  False)/0.01

	KLD_loss = -0.5*torch.sum(1 + sig_var - mu.pow(2) -sig_var.exp())
	return KLD_loss+ reconstruction_loss

def celeba_loss(recon_x,x,mu,sig_var):
	reconstruction_loss = F.mse_loss(recon_x,x,size_average =  False)/recon_x.size(0)/0.01

	KLD_element = mu.pow(2).add_(sig_var.exp()).mul_(-1).add_(1).add_(sig_var)
	KLD = torch.sum(KLD_element).mul_(0.5)
	return (KLD+ reconstruction_loss)