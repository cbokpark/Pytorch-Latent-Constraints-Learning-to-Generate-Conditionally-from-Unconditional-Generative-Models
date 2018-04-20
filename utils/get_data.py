import os 
import torch
import matplotlib.pyplot as plt 
from scipy.misc import resize 
from tqdm.tqdm import tqdm 
def celebA_data_preprocess(root ='/hdd1/celevA/data/CelebA_nocrop' ,save_root='/hdd1/celevA/data/CelebA_resize',resize=64):
	"""
		Preprocessing the celevA data set (resizing)
	"""



	if not os.path.isdir(save_root):
		os.mkdir(save_root)
	if not os.path.isdir(save_root + 'celebA'):
		os.mkdir(save_root+ 'celebA')
	img_list = os.listdir(root)

	for i in tqdm(range(len(img_list),desc='CelebA Preprocessing'):
		img = plt.imread(root+ img_list[i])
		img = imgresize(img,(resize,resize))
		plt.imsave(fname = save_root + 'celebA/'+img_list[i],arr=img)
	print ("[+] Finsish the CelebA Data set Preprocessing")
