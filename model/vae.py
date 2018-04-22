import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from sub_layer import Linear 

class Mnist_VAE(nn.Module):
	def __init__(self,input_dim,d_model,layer_num):
		self.d_model = d_model
		self.layer_num = layer_num
		self.input_dim = input_dim
		
		self.encoder = build_encoder(self.input_dim,self.d_model,self.layer_num)
		self.sig_layer = nn.SoftPlus()
		
		self.decoder = build_decoder(self.input_dim,self.d_model,self.layer_num)


	def build_encoder(self,input_dim,d_model,layer_num):
		encoder_layerList = []
		for i in range(layer_num):
			if i == 0 :
				encoder_layerList.append(Linear(input_dim*input_dim,d_model))
			else:
				encoder_layerList.append(Linear(d_model,d_model))
			encoder_layerList.append(nn.ReLU())
		encoder_layerList.append(Linear(d_model,2*d_model))
		return nn.Sequential(*encoder_layerList)

	def build_decoder(self,input_dim,d_model,layer_num):
		decoder_layerList = []
		for i in range(layer_num):
			decoder_layerList.append(Linear(d_model,d_model))
			decoder_layerList.append(nn.ReLU())
		
		decoder_layerList.append(Linear(d_model,input_dim*input_dim))
		
		return nn.Sequential(*decoder_layerList)
		
	def reparameterize(self,mu,sig_var):
		## need to understand
        if self.training:
            std = sig_var.exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
	
	def forward(self,x):
		encoder_out = self.encoder(x)
		sig_var , mu_var = encoder.chunk(2,dim=-1)
		sig_var = self.sig_layer(sig_var)
		z = reparameterize(mu_var,sig_var)
		output = self.decoder(z)

		return output,z,mu_var,sig_var

