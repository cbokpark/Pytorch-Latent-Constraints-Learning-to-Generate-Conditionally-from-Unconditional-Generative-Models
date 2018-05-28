import torch
import torch.nn.functional as F 

def loss_function(recon_x,x,mu,sig_var):
	reconstruction_loss = F.mse_loss(recon_x,x.view(-1,784),size_average =  False)/0.01

	KLD_element = mu.pow(2).add_(sig_var.pow(2)).mul_(-1).add_(1).add_(sig_var.pow(2).log())
	KLD = torch.sum(KLD_element).mul_(-0.5)
	return KLD+ reconstruction_loss

def celeba_loss(recon_x,x,mu,sig_var):
	reconstruction_loss = F.mse_loss(recon_x,x,size_average =  False)/recon_x.size(0)/0.1

	KLD_element = mu.pow(2).add_(sig_var.pow(2)).mul_(-1).add_(1).add_(sig_var.pow(2).log())
	KLD = torch.sum(KLD_element).mul_(-0.5)
	return (KLD+ reconstruction_loss)
class AC_loss:
	def __init__(self,lambda_dist,lambda_attr):
		self.lambda_attr = lambda_attr
		self.lambda_dist = lambda_dist
	def __real_loss(self,z,z_prime,sigvar,predict_d,grth_d):
		"""
			inputs:
				z : Batch Size * Z_dim 
				z_prime  : Batch Size * Z_dim
				sig_var : std  
		"""

		sum_variance = torch.sum(sigvar.pow(2),dim=-1)
		distance_penalty = torch.sum(F.mse_loss(z_prime,z,size_average=False,reduce =False)*sum_variance)
		real_loss = F.binary_cross_entropy(predict_d,grth_d)
		
		return real_loss

	#def __attr_loss(self,z,z_prime,sigvar,predict_d,grth_d)

