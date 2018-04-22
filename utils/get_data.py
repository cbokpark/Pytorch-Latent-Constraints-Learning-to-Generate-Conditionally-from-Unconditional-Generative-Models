import os 
import torch
import matplotlib.pyplot as plt 
from scipy.misc import imresize 

import torchvision.utils as utils 
import torchvision.datasets as vision_dsets
import torchvision.transforms as tv_transforms


from tqdm import tqdm 
def celebA_data_preprocess(root ='/hdd1/cheonbok_experiment/celevA/data/CelebA_nocrop/images/' ,save_root='/hdd1/cheonbok_experiment/celevA/data/CelebA_resize/',resize=64):
	"""
		Preprocessing the celevA data set (resizing)
	"""



	if not os.path.isdir(save_root):
		os.mkdir(save_root)
	if not os.path.isdir(save_root + 'celebA'):
		os.mkdir(save_root+ 'celebA')
	img_list = os.listdir(root)

	for i in tqdm(range(len(img_list)),desc='CelebA Preprocessing'):
		img = plt.imread(root+ img_list[i])
		img = imresize(img,(resize,resize))
		plt.imsave(fname = save_root + 'celebA/'+img_list[i],arr=img)
	print ("[+] Finished the CelebA Data set Preprocessing")

def MNIST_DATA(root='/hdd1/cheonbok_experiment/data/MNIST',train =True,transforms=None ,download =True,batch_size = 32,num_worker = 1):
	if transforms is None:
		transforms = transforms.ToTensor()
	print ("[+] Get the MNIST DATA")
	mnist_train = vision_dsets.MNIST(root = root, 
									train = True, 
									transforms = transforms,
									download = True)
	mnist_test = vision_dsets.MNIST(root = root,
									train = False, 
									transforms = tv_transforms.ToTensor(),
									download = True)
	trainDataLoader = utils.data.DataLoader(dataset = mnist_train,
											batch_size = batch_size,
											shuffle =True,
											num_worker = 1)

	testDataLoader = utils.data.DataLoader(dataset = mnist_test,
											batch_size = batch_size,
											shuffle = False,
											num_worker = 1)
	print ("[+] Finished loading data & Preprocessing")
	return mnist_train,mnist_test,trainDataLoader,testDataLoader

