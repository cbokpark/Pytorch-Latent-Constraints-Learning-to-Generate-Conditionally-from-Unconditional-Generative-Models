import torch
from torch.autograd import Variable 
from .sublayer import Linear
import torch.nn.functional as F
class Actor(nn.Module):
	def __init__(self,d_z,d_model,layer_num=4,num_label = 10):
		self.d_z = d_z
		self.d_model = d_model
		self.layer_num = layer_num
		layer_list = []
		for i in range(layer_num):
			if i == 0:
				layer_list.append(Linear(d_z,d_model))
			else:
				layer_list.append(Linear(d_model,d_model))
			layer_list.append(nn.ReLU)
		layer_list.append(Linear(d_model,d_model*2))

		self.fw_layer = nn.Sequential(*layer_list)
		self.gate = nn.Sigmoid()
		self.condition_layer = Linear(num_label,d_model)
	def forward(self,x,label= None):
		if label is not None:
			x = x + self.condition_layer(x)
		out = self.fw_layer(x)
		input_gate , dz = out.chunk(2,dim = -1)
		gate_value = self.gate(input_gate)
		return gate_value*dz

class Critic(nn.Module):
	def __init__(self,d_z,d_model,layer_num=4,num_label = 10):
		self.d_z = d_z
		self.d_model = d_model
		self.num_label = num_label
		layer_list = []
		for i in range(layer_num):
			if i == 0:
				layer_list.append(Linear(d_z,d_model))
			else:
				layer_list.append(Linear(d_model,d_model))
			layer_list.append(nn.ReLU)
		layer_list.append(Linear(d_model,1))
		self.fw_layer = nn.Sequential(*layer_list)
		self.condition_layer = Linear(num_label,d_model)
	def forward(self, x, label = None):
		if label is not None:
			x = x + self.condition_layer(x)
		out = self.fw_layer(x)
		return F.sigmoid(out)

