import numpy as np
import torch
import pdb
from torchvision.utils import save_image
import torch.optim as optim
class Trainer:
	# need to modify to inheritance version 
	def __init__(self,model,trainDataLoader,loss,epoch,metrics=None,resume=None,config=None,validDataLoader = None,device = 1, testDataLoader = None,train_logger =None,optimizer_type='Adam',lr=1e-3):
		#super(Trainer,self).__init__(model,loss,metrics,resume,config,train_logger)
		self.model = model
		self.trainDataLoader = trainDataLoader
		self.testDataLoader = testDataLoader
		self.validDataLoader = validDataLoader
		self.valid = True if self.validDataLoader is not None else False
		self.test = True if self.testDataLoader is not None else False
		self.device = device
		self.model.to(self.device)
		self.train_loss = 0

		self.epoch = epoch
		self.loss = loss
		self.start_epoch = 1
		self.with_cuda = torch.cuda.is_available()
		self.save_freq = 500
		self.total_iteration = 0 
		self.optimizer = getattr(optim, optimizer_type)(model.parameters(),lr=lr)
		self.valid_term = 10 
	def train(self):
		for epoch in range(self.start_epoch,self.epoch+1):
			result = self._train_epoch(epoch)
			self.get_sample(epoch)
			if epoch%self.valid_term == 0:
				self._test(epoch)
	def _train_epoch(self,epoch):

		self.model.train()
		train_loss = 0
		for batch_idx,(data,labels) in enumerate(self.trainDataLoader):
			data = data.to(self.device)
			self.optimizer.zero_grad()
			recon_batch,z,mu,log_sigma = self.model(data)
			loss = self.loss(recon_batch,data,mu,log_sigma)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
		print ("[+] Epoch:[{}/{}] train average loss :{}".format(epoch,self.epoch,train_loss))
			# print interval state 
						
	def _test(self,epoch):
		self.model.eval()
		test_loss = 0 
		with torch.no_grad():
			for i, (data,lebels) in enumerate(self.testDataLoader):
				data = data.cuda()
				recon_batch,z,mu,log_sigma = self.model(data)
				loss = self.loss(recon_batch,data,mu,log_sigma)
				test_loss += loss.item()
		print ("[+] Validation result {}".format(test_loss))
	def get_sample(self,epoch):
		with torch.no_grad():

			sample = torch.randn(64,400).to(self.device)
			sample = self.model.decoder(sample).cpu()
			save_image(sample.view(64,1,28,28),'results/sample_' + str(epoch) +'.png')

	def _eval_metric(self,output,target):
		raise NotImplementedError 

