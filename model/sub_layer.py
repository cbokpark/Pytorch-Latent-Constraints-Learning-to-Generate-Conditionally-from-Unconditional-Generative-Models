import torch
from torch.autograd import Variable 

class Linear(nn.Module):
	def __init__(self,input_dim,output_dim,**kwargs):
		self.linear = nn.Linear(kwargs)
		self.batch_norm = nn.batch_norm(output_dim)
	def forward(self,x):
		out = self.linear(x)
		return self.batch_norm(x)

	