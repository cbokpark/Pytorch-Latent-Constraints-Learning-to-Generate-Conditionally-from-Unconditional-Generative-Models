import numpy as np
import torch
import pdb
from torchvision.utils import save_image
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter 
import torch.nn.functional as F 
from tqdm import tqdm
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
				if self.data == 'MNIST':
					recon_batch = recon_batch.view(-1,1,28,28)
				save_image(recon_batch.cpu(),self.data+'_results/sample_train_' + str(epoch) +'.png')
				save_image(data.cpu(),self.data+'_results/grtruth_train_' + str(epoch) +'.png')
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
					if self.data == 'MNIST':
						recon_batch = recon_batch.view(-1,1,28,28)
					save_image(recon_batch.cpu(),self.data+'_results/sample_valid_' + str(epoch) +'.png')
					save_image(data.cpu(),self.data+'_results/grtruth_valid_' + str(epoch) +'.png')
				
		print ("[+] Validation result {}".format(test_loss))
	def get_sample(self,epoch):
		self.model.eval()
		with torch.no_grad():

			sample = torch.randn(64,self.d_model).to(self.device)
			out = self.model.decoder(sample)
			out = self.model.sigmoid(out)
			if self.data == 'MNIST':
				save_image(out.view(-1,1,28,28),self.data+'_results/sample_' + str(epoch) +'.png')
			else:	
				save_image(out.cpu(),self.data+'_results/sample_' + str(epoch) +'.png')
	def save_model(self,epoch):

		torch.save(self.model.state_dict(), './save_model/vae_model'+str(epoch)+'_'+self.data+'.path.tar')
	def _summary_wrtie(self,loss,epoch):
		self.tensorboad_writer.add_scalar('data/loss',loss,epoch)
		for name,param in self.model.named_parameters():
			self.tensorboad_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
			self.tensorboad_writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
	def _eval_metric(self,output,target):
		raise NotImplementedError 

class AC_Trainer:
	def __init__(self,vae_model,actor,real_critic,attr_critic,epoch,trainDataLoader,data,metrics=None,resume=None,config=None,validDataLoader = None,device = 0, testDataLoader = None,train_logger =None,optimizer_type='Adam',lr=1e-4):
		self.model = vae_model
		self.actor = actor
		self.real_critic = real_critic
		self.attr_critic = attr_critic
		#self.loss = loss # lossfunction class 
		self.epoch = epoch 

		self.trainDataLoader = trainDataLoader
		self.testDataLoader = testDataLoader
		self.validDataLoader = validDataLoader
		self.valid = True if self.validDataLoader is not None else False
		self.test = True if self.testDataLoader is not None else False
		self.device = device
		
		self.model.to(self.device)
		self.model.eval()
		self.actor.to(self.device)
		self.real_critic.to(self.device)
		self.attr_critic.to(self.device)

		self.d_model = self.model.d_model
		self.train_loss = 0

		self.iteration = 0 
		self.data = data
		self.tensorboad_writer = SummaryWriter()
		self.epoch = epoch
		#self.loss = loss
		
		self.start_epoch = 1
		self.with_cuda = torch.cuda.is_available()
		self.save_freq = 500
		self.total_iteration = 0 
		#self.gen_optimizer = getattr(optim, optimizer_type)(list(self.actor.parameters()) + list(self.critic.parameters()),lr=lr)
		self.actor_optim = getattr(optim, 'Adam')(self.actor.parameters(),lr=lr*3)
		self.real_optim = getattr(optim, 'Adam')(self.real_critic.parameters(),lr=lr*3)
		self.attr_optim = getattr(optim, optimizer_type)(self.attr_critic.parameters(),lr=lr*3)
		self.valid_term = 10 
		self._set_label_type()

	def train(self):
		for epoch in tqdm(range(self.start_epoch,self.epoch+1)):
			result = self._train_epoch(epoch)
			self.get_sample(epoch)
			if epoch%self.valid_term == 0:
				#self._test(epoch)
				self.get_sample(epoch)
				self.save_model(epoch)
			if (epoch+1) == 30:
				self.actor_optim.param_groups[0]['lr'] /= 10
				self.real_optim.param_groups[0]['lr'] /= 10
				print("learning rate change!")

			if (epoch+1) == 50:
				self.actor_optim.param_groups[0]['lr'] /= 10
				self.real_optim.param_groups[0]['lr'] /= 10
				print("learning rate change!")
		print ("[+] Finished Training Model")
	
	
	def _train_epoch(self,epoch):

		# Run Data Loader 
		# realism Constraints  pz~ is 0 E(q|x) ~ x
		# attr Constraints
		self.model.eval()
		self.actor.train()
		self.real_critic.train()
		self.attr_critic.train()
		train_loss = 0
		iteration = 0
		total_actor_loss = 0
		total_real_loss = 0
		total_distance_penalty = 0 
		actor_iteration = 0 
		for batch_idx,(data,labels) in enumerate(self.trainDataLoader):
			self.iteration += 1
			iteration += 1
			m_batchsize = data.size(0)

			real_data = torch.ones(m_batchsize,1)
			fake_data = torch.zeros(m_batchsize,1)
			fake_z = torch.randn(m_batchsize,self.d_model)
			fake_attr = self.fake_attr_generate(m_batchsize)

			real_data = real_data.to(self.device)
			fake_data = fake_data.to(self.device)
			labels = labels.to(self.device)
			fake_z = fake_z.to(self.device)
			fake_attr = fake_attr.to(self.device)
			data = data.to(self.device)
			with torch.no_grad():
				sig_var,mu,z = self.model.encode(data)
			
			z = self.reparameterize(mu,z)
			
			
			fake_z.requires_grad =True
			z.requires_grad = True
			labels.requires_grad =True
			fake_attr.requires_grad =True

			self.real_critic.zero_grad()
			if  np.random.rand(1) < 0.1:
				
				#input_data = torch.cat([z,fake_z],dim=0) 
				input_data = torch.cat([z,fake_z,z],dim=0) 
								

				input_attr = torch.cat([labels,labels,fake_attr],dim=0)
				#input_attr = torch.cat([labels,labels],dim=0)
				real_labels = torch.cat([real_data,fake_data,fake_data])
				
				#real_labels = torch.cat([real_data,fake_data])

				logit_out = self.real_critic(input_data,input_attr)
				critic_loss = F.binary_cross_entropy(logit_out,real_labels) 
			else:
				#pdb.set_trace()
				fake_z.requires_grad =True
				z_g = self.actor(fake_z,labels)		
				
				input_data = torch.cat([z,z_g,z],dim=0) 
				#input_data = torch.cat([z,z_g],dim=0) 

				input_attr = torch.cat([labels,labels,fake_attr],dim=0)
				#input_attr = torch.cat([labels,labels],dim=0)
				

				real_labels = torch.cat([real_data,fake_data,fake_data])
				#real_labels = torch.cat([real_data,fake_data])

				logit_out = self.real_critic(input_data,input_attr)

				critic_loss = F.binary_cross_entropy(logit_out,real_labels) 

			#print ("critic_loss : {}".format(critic_loss))
			critic_loss.backward()
			self.real_optim.step()
			if (batch_idx+1)%1000 == 0:
				self.d_critic_histogram(self.iteration)
			total_real_loss += critic_loss.item()

			if (batch_idx+1)%10 ==0:
				self.actor.zero_grad()	
				#actor_labels = self.re_allocate(labels)
				fake_z = torch.randn(m_batchsize,self.d_model)
				fake_z = fake_z.to(self.device)
				fake_z = self.re_allocate(fake_z)
				#actor_truth = self.re_allocate(real_data)

				actor_labels = labels
				fake_z = fake_z
				actor_truth = real_data

				actor_labels.requires_grad =True
				fake_z.requires_grad = True

				actor_g = self.actor(fake_z,actor_labels) 
				real_g = self.actor(z,actor_labels)
				zg_critic_out = self.real_critic(actor_g,actor_labels)
				zg_critic_real = self.real_critic(real_g,actor_labels)
				#fake_output = self.attr_critic(z_g) # prior 는 안써도 되는가? 애매하군. 
				weight_var = torch.mean(sig_var,0,True)

				distnace_penalty = torch.mean(torch.sum((1 + (actor_g-fake_z).pow(2)).log()*weight_var.pow(-2),1),0)
				#distnace_penalty = 0

				actor_loss = F.binary_cross_entropy(zg_critic_out,actor_truth,size_average=False)+ F.binary_cross_entropy(zg_critic_real,actor_truth,size_average=False)+distnace_penalty

				#actor_loss = actor_loss + distnace_penalty
				actor_loss.backward()
				total_actor_loss += actor_loss.item()
				self.actor_optim.step()
				total_distance_penalty += distnace_penalty.item()
				actor_iteration +=1
				if (actor_iteration%100) == 0 :
					self.d_actor_histogram(self.iteration)


			
		
			
			if batch_idx == 1:
				fake_z = torch.randn(m_batchsize,self.d_model)
				fake_z = fake_z.to(self.device)
				z_g = self.actor(fake_z,labels)
				z_g_recon = self.model.decode(z_g)
				prior_recon = self.model.decode(fake_z)
				data_recon = self.model.decode(z)
				if self.data == 'MNIST':
					data = data.view(-1,1,28,28)
					z_g_recon = z_g_recon.view(-1,1,28,28)
					data_recon = data_recon.view(-1,1,28,28)
					prior_recon = prior_recon.view(-1,1,28,28)
				save_image(z_g_recon.cpu(),self.data+'_results_ac/sample_z_g_train_' + str(epoch) +'.png')
				save_image(prior_recon.cpu(),self.data+'_results_ac/sample_prior_train_' + str(epoch) +'.png')	
				save_image(data_recon.cpu(),self.data+'_results_ac/sample_recon_train_' + str(epoch) +'.png')
				save_image(data.cpu(),self.data+'_results_ac/grtruth_train_' + str(epoch) +'.png')
		print ("distance penalty : {} , {} | critic_loss :{}".format(total_distance_penalty/actor_iteration,total_actor_loss/actor_iteration,total_real_loss/iteration))
				

		self._summary_wrtie(total_distance_penalty/actor_iteration,total_actor_loss/actor_iteration,total_real_loss/iteration,epoch)
		print ("[+] Epoch:[{}/{}] train actor average loss :{}".format(epoch,self.epoch,train_loss)) 
	def re_allocate(self,data):
		new_data = data.detach()
		new_data.requiers_grad = True
		return new_data
	def get_sample(self,epoch,data =None,labels =None):
		self.model.eval()
		self.actor.eval()
		self.real_critic.eval()
		self.attr_critic.eval()
		with torch.no_grad():
			for i in range(self.num_labels):
				test_labels = self.labels[i]
				test_labels = test_labels.expand(64,-1)

				sample = torch.randn(64,self.d_model).to(self.device)
				sample = self.actor(sample,test_labels)

				out = self.model.decoder(sample)
				out = self.model.sigmoid(out) #?? Amiguity Labels input? 
				if self.data == 'MNIST':
					save_image(out.view(-1,1,28,28),self.data+'_results_ac/sample_' + str(epoch)+'_class:'+str(i) +'.png')
				else:	
					save_image(out.cpu(),'results/sample_' + str(epoch) +'.png') 
	def _test(self,epoch):
		self.model.eval()
		self.actor.eval()
		self.real_critic.eval()
		self.attr_critic.eval()
		test_loss = 0 
		with torch.no_grad():
			for i, (data,lebels) in enumerate(self.testDataLoader):
				data = data.cuda()
				recon_batch,z,mu,log_sigma = self.model(data)
				
				loss = self.loss(recon_batch,data,mu,log_sigma)
				test_loss += loss.item()
				if i == 1:
					if self.data == 'MNIST':
						recon_batch = recon_batch.view(-1,1,28,28)
					save_image(recon_batch.cpu(),self.data+'_results/sample_valid_' + str(epoch) +'.png')
					save_image(data.cpu(),self.data+'_results/grtruth_valid_' + str(epoch) +'.png')

	def _set_label_type(self):
		
		self.labels = torch.eye(10)
		self.labels = self.labels.to(self.device)
		self.num_labels = self.labels.size(0) 
	
	def fake_attr_generate(self,batch_size,selection_index = None ):
		if self.data == 'MNIST':
			if selection_index is None:
				m = batch_size
				selection = np.random.randint(self.num_labels,size=m)
				selection = torch.from_numpy(selection).to(self.device)
				fake_attr = torch.index_select(self.labels,0,selection)
			else:
				selection_index = selection_index.cuda()
				fake_attr = torch.index_select(self.labels,0,selection_index)
		else:
			selection_start = np.random.randint(self.trainDataLoader.dataset.num_images-batch_size-1)
			selection_data = self.trainDataLoader.dataset.train_dataset[selection_start:selection_start+batch_size]
			fake_attr = []
			for (name,labels) in selection_data:
				fake_attr.append(labels)
			fake_attr = torch.FloatTensor(fake_attr)
			
		return fake_attr


	def reparameterize(self,mu,sig_var):

		std = sig_var # need to check sig_var is log (sigma^2)
		eps = std.data.new(std.size()).normal_(std=1)
		return eps.mul(std).add_(mu)
	def _summary_wrtie(self,distance_penalty,loss,real_loss,epoch):
		self.tensorboad_writer.add_scalar('data/loss',loss,epoch) # need to modify . We use four loss value . 
		self.tensorboad_writer.add_scalar('data/distance_penalty',distance_penalty,epoch) # need to modify . We use four loss value . 
		self.tensorboad_writer.add_scalar('data/discriminator_loss',real_loss,epoch)
		#for name,param in self.actor.named_parameters(): #actor
		#	self.tensorboad_writer.add_histogram('actor/'+name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
		#	self.tensorboad_writer.add_histogram('actor/'+name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
		#for name,param in self.real_critic.named_parameters(): #actor
		#	self.tensorboad_writer.add_histogram('real_critic/'+name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
		#	self.tensorboad_writer.add_histogram('real_critic/'+name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
		#for name,param in self.attr_critic.named_parameters(): #actor
		#	self.tensorboad_writer.add_histogram('attr_critic/'+name, param.clone().cpu().data.numpy(), epoch,bins='sturges')
		#	self.tensorboad_writer.add_histogram('attr_critic/'+name+'/grad', param.grad.clone().cpu().data.numpy(), epoch,bins='sturges')
	def d_critic_histogram(self,iteration):
		for name,param in self.real_critic.named_parameters(): #actor
			self.tensorboad_writer.add_histogram('real_critic/'+name, param.clone().cpu().data.numpy(), iteration,bins='sturges')
			self.tensorboad_writer.add_histogram('real_critic/'+name+'/grad', param.grad.clone().cpu().data.numpy(), iteration,bins='sturges')
	def d_actor_histogram(self,iteration):
		for name,param in self.actor.named_parameters(): #actor
			self.tensorboad_writer.add_histogram('actor/'+name, param.clone().cpu().data.numpy(), iteration,bins='sturges')
			self.tensorboad_writer.add_histogram('actor/'+name+'/grad', param.grad.clone().cpu().data.numpy(), iteration,bins='sturges')
	def save_model(self,epoch):
		torch.save(self.actor.state_dict(), './save_model/actor_model'+str(epoch)+'_'+self.data+'.path.tar')
		torch.save(self.real_critic.state_dict(), './save_model/real_d_model'+str(epoch)+'_'+self.data+'.path.tar')
		torch.save(self.attr_critic.state_dict(), './save_model/attr_d_model'+str(epoch)+'_'+self.data+'.path.tar')
	def load_vae(self,path):
		
		print ("[+] Load pre-trained VAE model")
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint)
		