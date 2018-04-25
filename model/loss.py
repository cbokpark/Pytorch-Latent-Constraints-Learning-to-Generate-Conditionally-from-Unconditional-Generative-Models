import torch
import torch.nn.functional as F 

def loss_function(x,recon_x,mu,sig_var):
	Reconstruction_loss = F.binary_cross_entropy(recon_x,x.view(-1,784),size_average =  False)

	KLD_loss = -0.5*torch.sum(1 + sig_var - mu.pow(2) -sig_var.exp())
	return KLD_loss+ Reconstruction_loss
	