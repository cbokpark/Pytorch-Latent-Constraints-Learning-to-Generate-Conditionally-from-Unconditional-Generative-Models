import argparse 
import numpy 
from model.trainer import Trainer,AC_Trainer 
from model.loss import loss_function,celeba_loss


from model.vae import Mnist_VAE,Celeba_VAE
from utils.get_data import MNIST_DATA ,Celeba_DATA
from model import Actor,Critic
import torch

def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--gpu_num', type = int, default = None)
	parser.add_argument('--data',type =str,required=True)
	parser.add_argument('--num_epoch',type=int,default =50)
	parser.add_argument('--batch_size',type=int,default =128)
	parser.add_argument('--tensorboard_dirs',type=str,default ='./run')
	parser.add_argument('--train_id',type=str , default = 'my_model')
	parser.add_argument('--gpu_accelerate',action='store_true')
	parser.add_argument('--image_dir',type = str , default = './data/CelebA_nocrop/images')
	parser.add_argument('--attr_path',type = str ,default = './data/list_attr_celeba.txt')
	parser_config = parser.parse_args()
	print (parser_config)
	if parser_config.gpu_num is not None :
		torch.cuda.set_device(parser_config.gpu_num)

	if parser_config.gpu_accelerate:
		torch.backends.cudnn.benchmark = True
	if parser_config.gpu_num == -1:
		device = 'cpu'
	else:
		device = parser_config.gpu_num
	if parser_config.vae_mode:

		if parser_config.data == 'MNIST':
			trainDset,testDset,trainDataLoader,testDataLoader= MNIST_DATA(batch_size = parser_config.batch_size )
			model = Mnist_VAE(input_dim= 28*28 ,layer_num= 4, d_model=400)
			trainer = Trainer(model=model,
							loss = loss_function,
							epoch = parser_config.num_epoch,
							trainDataLoader=trainDataLoader,
							testDataLoader=testDataLoader)

		elif parser_config.data == 'celeba':
			trainDset,testDset,trainDataLoader,testDataLoader = Celeba_DATA(celeba_img_dir=parser_config.image_dir ,attr_path=parser_config.attr_path,batch_size = parser_config.batch_size,image_size=64,celeba_crop_size=128)
			model = Celeba_VAE(128,d_model=1024,layer_num=3)
			trainer = Trainer(model=model,
							loss = celeba_loss,
							epoch = parser_config.num_epoch,
							trainDataLoader=trainDataLoader,
							testDataLoader=testDataLoader)

		else:
			raise NotImplementedError
	else:

		if parser_config.data ==  'celeba':
			trainDset,testDset,trainDataLoader,testDataLoader = Celeba_DATA(celeba_img_dir=parser_config.image_dir ,attr_path=parser_config.attr_path,batch_size = parser_config.batch_size,image_size=64,celeba_crop_size=128)
			model = Celeba_VAE(128,d_model=1024,layer_num=3)
			actor = 
			real_critic =
			attr_critic =
			actrainer = AC_Trainer(model=model,
							loss = celeba_loss,
							epoch = parser_config.num_epoch,
							trainDataLoader=trainDataLoader,
							testDataLoader=testDataLoader)
			trainer.load_vae('')			
	print ("[+] Train model start")
	trainer.train()

if __name__ == '__main__':
	main()