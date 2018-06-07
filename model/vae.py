import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from .sub_layer import Linear,View
import pdb

class Celeba_VAE(nn.Module):
	def __init__(self,input_size=128,d_model=1024,layer_num=3):
		super(Celeba_VAE,self).__init__()
		self.d_model = d_model
		self.layer_num = layer_num
		
		self.encoder = self.build_encoder(self.d_model,self.layer_num)
		self.sig_layer = nn.Softplus()
		
		self.decoder = self.build_decoder(self.d_model,self.layer_num)
		self.sigmoid = nn.Sigmoid()


	def build_encoder(self,d_model,layer_num):
		encoder_layerList = []

		
		encoder_layerList.append(nn.Conv2d(in_channels = 3,out_channels = 256,kernel_size = 5,stride =2,padding=1))
		#encoder_layerList.append(nn.BatchNorm2d(256))
		encoder_layerList.append(nn.ReLU())
		encoder_layerList.append(nn.Conv2d(in_channels = 256 ,out_channels = 256*2 , kernel_size = 5,stride =2,padding=1))
		#encoder_layerList.append(nn.BatchNorm2d(256*2))
		encoder_layerList.append(nn.ReLU())
		encoder_layerList.append(nn.Conv2d(in_channels=512 , out_channels = 1024, kernel_size = 3,stride =2,padding=1))
		#encoder_layerList.append(nn.BatchNorm2d(512*2))
		encoder_layerList.append(nn.ReLU())
		encoder_layerList.append(nn.Conv2d(in_channels = 1024,out_channels = 2048,kernel_size = 3,stride =2,padding=1))		
		#encoder_layerList.append(nn.BatchNorm2d(1024*2))
		encoder_layerList.append(nn.ReLU())
		encoder_layerList.append(View())
		encoder_layerList.append(nn.Linear(4*4*2048,2048))
		return nn.Sequential(*encoder_layerList)

	def build_decoder(self,d_model,layer_num):
		decoder_layerList = []
		decoder_layerList
		
		decoder_layerList.append(nn.Linear(d_model,2048*4*4))
		decoder_layerList.append(View([2048,4,4]))
		decoder_layerList.append(nn.ConvTranspose2d(2048,1024,3,stride=2 ,padding =1 ,output_padding=1))
		#decoder_layerList.append(nn.BatchNorm2d(1024))
		decoder_layerList.append(nn.ReLU())
		decoder_layerList.append(nn.ConvTranspose2d(1024,512,3,stride=2 ,padding =1 ,output_padding=0))
		#decoder_layerList.append(nn.BatchNorm2d(512))
		decoder_layerList.append(nn.ReLU())
		decoder_layerList.append(nn.ConvTranspose2d(512,256,5,stride=2,padding=1 ,output_padding=0))
		#decoder_layerList.append(nn.BatchNorm2d(256))
		decoder_layerList.append(nn.ReLU())
		decoder_layerList.append(nn.ConvTranspose2d(256,3,5,stride=2 ,padding = 1 ,output_padding =1))
		return nn.Sequential(*decoder_layerList)
		
	def reparameterize(self,mu,sig_var):
		## need to understand
		if self.training:
			std = sig_var # need to check sig_var is log (sigma^2)
			eps = std.data.new(std.size()).normal_(std=1)
			return eps.mul(std).add_(mu)
		else:
			return mu
	def encode(self,x):
		encoder_out = self.encoder(x)

		sig_var , mu_var = encoder_out.chunk(2,dim=-1)

		sig_var = self.sig_layer(sig_var)
		z = self.reparameterize(mu_var,sig_var)
		return sig_var,mu_var,z 
	def decode(self,z):
		output = self.decoder(z)
		output = self.sigmoid(output)
		return output

	def forward(self,x):
		encoder_out = self.encoder(x)
		sig_var , mu_var = encoder_out.chunk(2,dim=-1)

		sig_var = self.sig_layer(sig_var)
		z = self.reparameterize(mu_var,sig_var)

		output = self.decoder(z)
		output = self.sigmoid(output)
		return output,z,mu_var,sig_var



class Mnist_VAE(nn.Module):
	def __init__(self,input_dim=28*28,d_model=1024,layer_num=3):
		super(Mnist_VAE,self).__init__()
		self.d_model = d_model
		self.layer_num = layer_num
		self.input_dim = input_dim
		
		self.encoder = self.build_encoder(self.input_dim,self.d_model,self.layer_num)
		self.sig_layer = nn.Softplus()
		
		self.decoder = self.build_decoder(self.input_dim,self.d_model,self.layer_num)

		self.sigmoid = nn.Sigmoid()
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
			std = sig_var # need to check sig_var is log (sigma^2)
			eps = std.data.new(std.size()).normal_(std=1)
			return eps.mul(std).add_(mu)
		else:
			return mu
	def encode(self,x):
		x = x.view(-1,28*28)
		encoder_out = self.encoder(x)
		sig_var , mu_var = encoder_out.chunk(2,dim=-1)

		sig_var = self.sig_layer(sig_var)
		z = self.reparameterize(mu_var,sig_var)
		return sig_var,mu_var,z 

	def decode(self,z):
		output = self.decoder(z)
		output = self.sigmoid(output)
		return output
		
	def forward(self,x):
		x = x.view(-1,28*28)
		encoder_out = self.encoder(x)
		sig_var , mu_var = encoder_out.chunk(2,dim=-1)

		sig_var = self.sig_layer(sig_var)
		z = self.reparameterize(mu_var,sig_var)
		output = self.decoder(z)
		output = self.sigmoid(output)

		return output,z,mu_var,sig_var


