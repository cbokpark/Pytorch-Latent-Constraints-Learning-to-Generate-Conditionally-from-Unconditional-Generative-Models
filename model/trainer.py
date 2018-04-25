import numpy as np
import torch
from .basetrainer import BaseTrain
from torchvision.utils import save_image
class Trainer(BaseTrain):
	def __init__(self,model,loss,metrics,resume,config,trainDataloader,validDataloader = None,device = 1, testDataLoader = None,train_logger =None):
		super(Trainer,self).__init__(model,loss,metrics,resume,config,train_logger)
		self.trainDataloader = trainDataloader
		self.testDataLoader = testDataLoader
		self.validDataloader = validDataloader
		self.valid = True if self.validDataloader is not None else False
		self.test = True if self.testDataLoader is not None else False
		self.device = device
		self.model.to(self.device)
		self.train_loss = 0 
	def _train_epoch(self,epoch):

		self.model.train()
		for batch_idx,(data,labels) in enumerate(self.trainDataloader):
			data = data.to(self.device)
			self.optimizer.zero_grad()
			recon_batch,mu,log_sigma = model(data)
			loss = self.loss(recon_batch,data,mu,log_sigma)
			loss.backward()
			self.optimizer.step()
			self.train_loss += loss.data[0]

			# print interval state 
						
	def test(self.epoch):
		self.model.eval()
		test_loss = 0 
		with torch.no_grad():
			for i, (data,lebels) in enumerate(self.validDataloader):
				data = data.to(self.device)
				recon_batch,mu,log_sigma = self.model(data)
				loss = self.loss(recon_batch,data,mu,log_sigma)
				test_loss += test_loss.data[0]
		print ("[+] Validation result {}".format(test_loss))
	def get_sample(epoch):
		with torch.no_grad():
			sample = torch.randn(64,20).to(self.device)
			sample = model.decode(sample).cpu()
			save_image(sample.view(64,1,28,28),+'results/sample' + str(epoch) +'.png')

	def _eval_metric(self,output,target):
		raise NotImplementedError 

