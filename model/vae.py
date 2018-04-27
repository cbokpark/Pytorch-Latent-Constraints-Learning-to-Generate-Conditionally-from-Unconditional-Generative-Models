import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from .sub_layer import Linear 
import pdb
class Mnist_VAE(nn.Module):
	def __init__(self,input_dim=28*28,d_model=400,layer_num=3):
		super(Mnist_VAE,self).__init__()
		self.d_model = d_model
		self.layer_num = layer_num
		self.input_dim = input_dim
		
		self.encoder = self.build_encoder(self.input_dim,self.d_model,self.layer_num)
		self.sig_layer = nn.Softplus()
		
		self.decoder = self.build_decoder(self.input_dim,self.d_model,self.layer_num)


	def build_encoder(self,input_dim,d_model,layer_num):
		encoder_layerList = []
		for i in range(layer_num):
			if i == 0 :
				encoder_layerList.append(nn.Linear(input_dim,d_model))
			else:
				encoder_layerList.append(nn.Linear(d_model,d_model))
			encoder_layerList.append(nn.ReLU())
		encoder_layerList.append(nn.Linear(d_model,2*d_model))
		return nn.Sequential(*encoder_layerList)

	def build_decoder(self,input_dim,d_model,layer_num):
		decoder_layerList = []
		for i in range(layer_num):
			decoder_layerList.append(nn.Linear(d_model,d_model))
			decoder_layerList.append(nn.ReLU())
		
		decoder_layerList.append(nn.Linear(d_model,input_dim))
		
		return nn.Sequential(*decoder_layerList)
		
	def reparameterize(self,mu,sig_var):
		## need to understand
		if self.training:
			std = sig_var.mul(0.5).exp_() # need to check sig_var is log (sigma^2)
			eps = std.data.new(std.size()).normal_()
			return eps.mul(std).add_(mu)
		else:
			return mu
	
	def forward(self,x):
		x = x.view(-1,28*28)
		encoder_out = self.encoder(x)
		sig_var , mu_var = encoder_out.chunk(2,dim=-1)

		sig_var = self.sig_layer(sig_var)
		z = self.reparameterize(mu_var,sig_var)
		output = self.decoder(z)

		return output,z,mu_var,sig_var


