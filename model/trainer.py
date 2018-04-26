import numpy as np
import torch
from .basetrainer import BaseTrain
from torchvision.utils import save_image
import torch.optim as optim
class Trainer:
	# need to modify to inheritance version 
	def __init__(self,model,loss,metrics,resume,config,trainDataloader,validDataloader = None,device = 1, testDataLoader = None,train_logger =None,optimizer_type='Adam',lr=1e-3):
		super(Trainer,self).__init__(model,loss,metrics,resume,config,train_logger)
		self.trainDataloader = trainDataloader
		self.testDataLoader = testDataLoader
		self.validDataloader = validDataloader
		self.valid = True if self.validDataloader is not None else False
		self.test = True if self.testDataLoader is not None else False
		self.device = device
		self.model.to(self.device)
		self.train_loss = 0


		self.start_epoch = 1
		self.with_cuda = config['cuda'] and torch.cuda.is_available()
        self.save_freq = 500
        self.total_iteration = 0 
        self.optimizer = getattr(optim, optimizer_type)(model.parameters(),lr=lr)
        self.valid_term = 10 
	def train(self):
		for epoch in range(self.start_epoch,self.epoch+1):
			result = self._train_epoch(epoch)
			self.get_sample(epoch)
			if epoch%self.valid_term == 0:
				test(epoch)
	def _train_epoch(self,epoch):

		self.model.train()
		train_loss = 0
		for batch_idx,(data,labels) in enumerate(self.trainDataloader):
			data = data.to(self.device)
			self.optimizer.zero_grad()
			recon_batch,mu,log_sigma = model(data)
			loss = self.loss(recon_batch,data,mu,log_sigma)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.data[0]
		print ("[+] Epoch:[{}/{}] train average loss :{}".format(epoch,self.epoch,train_loss))
			# print interval state 
						
	def test(self,epoch):
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
			for i in range(10):
				sample = torch.randn(64,20).to(self.device)
				sample = model.decode(sample).cpu()
				save_image(sample.view(64,1,28,28),+'results/sample' + str(epoch) +'_'+i+'.png')

	def _eval_metric(self,output,target):
		raise NotImplementedError 

