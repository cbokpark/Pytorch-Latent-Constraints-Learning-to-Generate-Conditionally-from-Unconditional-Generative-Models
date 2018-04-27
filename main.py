import argparse 
import numpy 
from model.trainer import Trainer 
from model.loss import loss_function

from model.vae import Mnist_VAE
from utils.get_data import MNIST_DATA 
import torch

def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--gpu_num', type = int, default = None)
	parser.add_argument('--data',type =str,required=True)
	parser.add_argument('--num_epoch',type=int,default =50)
	parser.add_argument('--batch_size',type=int,default =64)
	parser.add_argument('--tensorboard_dirs',type=str,default ='./run')
	parser.add_argument('--train_id',type=str , default = 'my_model')
	parser.add_argument('--gpu_accelerate',action='store_true')
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
	if parser_config.data == 'MNIST':
		trainDset,testDset,trainDataLoader,testDataLoader= MNIST_DATA(batch_size = parser_config.batch_size )
		model = Mnist_VAE(input_dim= 28*28 ,layer_num= 4, d_model=400)
		trainer = Trainer(model=model,
						loss = loss_function,
						epoch = parser_config.num_epoch,
						trainDataLoader=trainDataLoader,
						testDataLoader=testDataLoader)
	else:
		raise NotImplementedError
	print ("[+] Train model start")
	trainer.train()

if __name__ == '__main__':
	main()