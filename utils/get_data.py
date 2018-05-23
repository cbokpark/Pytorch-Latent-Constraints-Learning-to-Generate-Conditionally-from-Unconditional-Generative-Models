import os 
import torch
import matplotlib.pyplot as plt 
from scipy.misc import imresize 

from PIL import Image
import random 

import torchvision.utils as utils 
from torch.utils import data
import torchvision.datasets as vision_dsets
import torchvision.transforms as T


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
		transforms = T.ToTensor()
	print ("[+] Get the MNIST DATA")
	mnist_train = vision_dsets.MNIST(root = root, 
									train = True, 
									transform = transforms,
									download = True)
	mnist_test = vision_dsets.MNIST(root = root,
									train = False, 
									transform = T.ToTensor(),
									download = True)
	trainDataLoader = data.DataLoader(dataset = mnist_train,
									batch_size = batch_size,
									shuffle =True,
									num_workers = 1)

	testDataLoader = data.DataLoader(dataset = mnist_test,
									batch_size = batch_size,
									shuffle = False,
									num_workers = 1)
	print ("[+] Finished loading data & Preprocessing")
	return mnist_train,mnist_test,trainDataLoader,testDataLoader


class CelebA(data.Dataset):
	"""Dataset class for the CelebA dataset."""

	def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
		"""Initialize and preprocess the CelebA dataset."""
		self.image_dir = image_dir
		self.attr_path = attr_path
		self.selected_attrs = selected_attrs
		self.transform = transform
		self.mode = mode
		self.train_dataset = []
		self.test_dataset = []
		self.attr2idx = {}
		self.idx2attr = {}
		self.preprocess()

		if mode == 'train':
			self.num_images = len(self.train_dataset)
		else:
			self.num_images = len(self.test_dataset)

	def preprocess(self):
		"""Preprocess the CelebA attribute file."""
		lines = [line.rstrip() for line in open(self.attr_path, 'r')]
		all_attr_names = lines[1].split()
		for i, attr_name in enumerate(all_attr_names):
			self.attr2idx[attr_name] = i
			self.idx2attr[i] = attr_name

		lines = lines[2:]
		random.seed(1234)
		random.shuffle(lines)
		for i, line in enumerate(lines):
			split = line.split()
			filename = split[0]
			values = split[1:]

			label = []
			for attr_name in self.selected_attrs:
				idx = self.attr2idx[attr_name]
				label.append(values[idx] == '1')

			if (i+1) < 2000:
				self.test_dataset.append([filename, label])
			else:
				self.train_dataset.append([filename, label])

		print('[+]Finished preprocessing the CelebA dataset...')

	def __getitem__(self, index):
		"""Return one image and its corresponding attribute label."""
		dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
		filename, label = dataset[index]
		image = Image.open(os.path.join(self.image_dir, filename))
		return self.transform(image), torch.FloatTensor(label)

	def __len__(self):
		"""Return the number of images."""
		return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
			   batch_size=16, dataset='CelebA', mode='train', num_workers=1):
	"""Build and return a data loader."""
	transform = []
	if mode == 'train':
		transform.append(T.RandomHorizontalFlip())
	transform.append(T.CenterCrop(crop_size))
	transform.append(T.Resize(image_size))
	transform.append(T.ToTensor())
	#transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = T.Compose(transform)

	if dataset == 'CelebA':
		dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
	elif dataset == 'RaFD':
		dataset = ImageFolder(image_dir, transform)

	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=(mode=='train'),
								  num_workers=num_workers)
	return data_loader,dataset

def Celeba_DATA(celeba_img_dir ,attr_path,image_size=128,celeba_crop_size=178,selected_attrs=None,batch_size = 32,num_worker = 1):
	


	if selected_attrs is None:
		selected_attrs = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes', 'Bangs', 
						'Big_Lips','Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
						 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
						'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
						'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
						'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
	trainDataLoader,trainData = get_loader(celeba_img_dir,attr_path,selected_attrs,
								celeba_crop_size,image_size,batch_size,
								'CelebA','train',num_worker)
	testDataLoader,testData = get_loader(celeba_img_dir,attr_path,selected_attrs,
								celeba_crop_size,image_size,batch_size,
								'CelebA','test',num_worker)

	return trainData,testData,trainDataLoader,testDataLoader