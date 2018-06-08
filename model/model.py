import torch
import torch.nn as nn
from .sub_layer import Linear
import torch.nn.functional as F
import pdb
class Actor(nn.Module):
	def __init__(self,d_z,d_model=2048,layer_num=4,num_label = 10,condition_mode = True):
		super(Actor,self).__init__()
		self.d_z = d_z
		self.d_model = d_model
		self.layer_num = layer_num
		self.condition_mode = condition_mode
		layer_list = []
		for i in range(layer_num):
			if i == 0:
				if condition_mode:
					input_dim = d_z + d_model
				else:
					input_dim = d_z 
				layer_list.append(Linear(input_dim,d_model))
			else:
				layer_list.append(Linear(d_model,d_model))
			layer_list.append(nn.ReLU())
		layer_list.append(Linear(d_model,self.d_z*2))

		self.fw_layer = nn.Sequential(*layer_list)
		self.gate = nn.Sigmoid()
		if condition_mode:
			self.num_label = num_label
			self.condition_layer = Linear(num_label,d_model)

	def forward(self,x,label):
		original_x = x
		
		x = torch.cat((x,self.condition_layer(label)),dim = -1)
		out = self.fw_layer(x)
		input_gate , dz = out.chunk(2,dim = -1)
		gate_value = self.gate(input_gate)
		new_z = (1-gate_value)*original_x + gate_value*dz
		return new_z

class Critic(nn.Module):
	def __init__(self,d_z,d_model,layer_num=4,num_labels = None,num_output = 1,condition_mode = True):
		super(Critic,self).__init__()
		self.d_z = d_z
		self.d_model = d_model
		self.condition_mode = condition_mode
		layer_list = []
		for i in range(layer_num):
			if i == 0:
				if condition_mode:
					input_dim = d_z + d_model
				else:
					input_dim = d_z 
				layer_list.append(Linear(input_dim,d_model))
			else:
				layer_list.append(Linear(d_model,d_model))
			layer_list.append(nn.ReLU())
		layer_list.append(Linear(d_model,num_output))
		self.fw_layer = nn.Sequential(*layer_list)

		if condition_mode:
			self.num_labels = num_labels
			self.condition_layer = Linear(num_labels,d_model)
	def forward(self, x, label = None):
		
		if self.condition_mode:
			x = torch.cat((x,self.condition_layer(label)),dim = -1)
		out = self.fw_layer(x)
		return F.sigmoid(out)

