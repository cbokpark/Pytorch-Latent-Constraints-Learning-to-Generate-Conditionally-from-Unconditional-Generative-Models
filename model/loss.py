import torch
import torch.nn.functional as F 
import pdb
def loss_function(recon_x,x,mu,sig_var):
	reconstruction_loss = F.mse_loss(recon_x,x.view(-1,784),size_average =  False)

	KLD_loss = -0.5*torch.sum(1 + sig_var - mu.pow(2) -sig_var.exp())
	return KLD_loss+ reconstruction_loss
	