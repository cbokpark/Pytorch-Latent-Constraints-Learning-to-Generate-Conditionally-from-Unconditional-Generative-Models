import numpy as np
import torch
import pdb
from torchvision.utils import save_image
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter 

class Trainer:
	# need to modify to inheritance version 
	def __init__(self,model,trainDataLoader,loss,epoch,data = 'celeba' ,metrics=None,resume=None,config=None,validDataLoader = None,device = 0, testDataLoader = None,train_logger =None,optimizer_type='Adam',lr=1e-3):
		#super(Trainer,self).__init__(model,loss,metrics,resume,config,train_logger)
		self.model = model
		self.trainDataLoader = trainDataLoader
		self.testDataLoader = testDataLoader
		self.validDataLoader = validDataLoader
		self.valid = True if self.validDataLoader is not None else False
		self.test = True if self.testDataLoader is not None else False
		self.device = device
		self.model.to(self.device)
		self.d_model = self.model.d_model
		self.train_loss = 0

		self.data = data
		self.tensorboad_writer = SummaryWriter()
		self.epoch = epoch
		self.loss = loss
		self.start_epoch = 1
		self.with_cuda = torch.cuda.is_available()
		self.save_freq = 500
		self.total_iteration = 0 
		self.optimizer = getattr(optim, optimizer_type)(self.model.parameters(),lr=lr)
		self.valid_term = 10 
	def train(self):
		for epoch in range(self.start_epoch,self.epoch+1):
			result = self._train_epoch(epoch)
			self.get_sample(epoch)
			if epoch%self.valid_term == 0:
				self._test(epoch)
				self.save_model(epoch)
		print ("[+] Finished Training Model")
		
		
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
			if batch_idx == 1:
				save_image(recon_batch.cpu(),'results/sample_train_' + str(epoch) +'.png')
				save_image(data.cpu(),'results/grtruth_train_' + str(epoch) +'.png')
		self._summary_wrtie(train_loss,epoch)
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
				if i == 1:
					save_image(recon_batch.cpu(),'results/sample_valid_' + str(epoch) +'.png')
					save_image(data.cpu(),'results/grtruth_valid_' + str(epoch) +'.png')
				
		print ("[+] Validation result {}".format(test_loss))
	def get_sample(self,epoch):
		self.model.eval()
		with torch.no_grad():

			sample = torch.randn(64,self.d_model).to(self.device)
			out = self.model.decoder(sample)
			out = self.model.sigmoid(out)
			if self.data == 'mnist':
				save_image(sample.view(64,1,28,28),'results/sample_' + str(epoch) +'.png')
			else:	
				save_image(out.cpu(),'results/sample_' + str(epoch) +'.png')
	def save_model(self,epoch):

		torch.save(self.model.state_dict(), './save_model/vae_model'+str(epoch))
	def _summary_wrtie(self,loss,epoch):
		self.tensorboad_writer.add_scalar('data/loss',loss,epoch)
		for name,param in self.model.named_parameters():
			self.tensorboad_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
			self.tensorboad_writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
	def _eval_metric(self,output,target):
		raise NotImplementedError 

class AC_Trainer:
	def __init__(self,vae_model,actor,critic,loss,epoch,metrics=None,resume=None,config=None,validDataLoader = None,device = 0, testDataLoader = None,train_logger =None,optimizer_type='Adam',lr=1e-3):
		self.vae_model = vae_model
		self.actor = actor
		self.critic = critic
		self.loss = loss # lossfunction class 
		self.epoch = epoch 

		self.trainDataLoader = trainDataLoader
		self.testDataLoader = testDataLoader
		self.validDataLoader = validDataLoader
		self.valid = True if self.validDataLoader is not None else False
		self.test = True if self.testDataLoader is not None else False
		self.device = device
		self.vae_model.to(self.device)
		self.vae_model.eval()

		self.d_model = self.model.d_model
		self.train_loss = 0

		self.iteration = 0 
		self.data = data
		self.tensorboad_writer = SummaryWriter()
		self.epoch = epoch
		self.loss = loss
		self.start_epoch = 1
		self.with_cuda = torch.cuda.is_available()
		self.save_freq = 500
		self.total_iteration = 0 
		self.optimizer = getattr(optim, optimizer_type)(list(self.actor.parameters()) + list(self.critic.parameters()),lr=lr)
		self.valid_term = 10 

	def train(self):
		for epoch in range(self.start_epoch,self.epoch+1):
			result = self._train_epoch(epoch)
			self.get_sample(epoch)
			if epoch%self.valid_term == 0:
				self._test(epoch)
				self.save_model(epoch)
		print ("[+] Finished Training Model")
	
	def realism(self):
		raise NotImplementedError
	def transformation(self):
		raise NotImplementedError
	def _train_epoch(self,epoch):

		raise NotImplementedError
	def get_sample(self,labels):
		self.actor.eval()
		self.critic.eval()
		with torch.no_grad():
			sample = torch.randn(64,self.d_model).to(self.device)
			out = self.model.decoder(sample)
			out = self.actor(out,labels) #?? Amiguity Labels input? 
			if self.data == 'mnist':
				save_image(out.view(64,1,28,28),'results/sample_' + str(epoch) +'.png')
			else:	
				save_image(out.cpu(),'results/sample_' + str(epoch) +'.png') 
	def _test(self,epoch):
		raise NotImplementedError


	def _summary_wrtie(self,loss,epoch):
		self.tensorboad_writer.add_scalar('data/loss',loss,epoch) # need to modify . We use four loss value . 
		for name,param in self.actor.named_parameters(): #actor
			self.tensorboad_writer.add_histogram('actor/'+name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
			self.tensorboad_writer.add_histogram('actor/'+name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
		for name,param in self.crtic.named_parameters(): #actor
			self.tensorboad_writer.add_histogram('critic/'+name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
			self.tensorboad_writer.add_histogram('critic/'+name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
	def _save_model(self,epoch):
		raise NotImplementedError
