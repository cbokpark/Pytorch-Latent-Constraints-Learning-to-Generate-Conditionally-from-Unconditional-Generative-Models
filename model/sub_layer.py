import torch
import torch.nn as nn 

class Linear(nn.Module):
	def __init__(self,input_dim,output_dim,bias = True):
		super(Linear,self).__init__()
		self.linear = nn.Linear(input_dim,output_dim,bias=bias)
		self.batch_norm = nn.BatchNorm1d(output_dim)
	def forward(self,x):
		out = self.linear(x)
		return self.batch_norm(out)

	