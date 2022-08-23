import torch
from torch.utils import data
from nilearn.input_data import NiftiMasker
import nibabel as nib
import numpy as np
import os

from utils.transform import inverse_transform

class LoadHCP(data.Dataset):
	def __init__(self, task="EMOTION", load_num=None,
				mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
				HCP_path="/home/public/ExperimentData/HCP900/HCP_data/SINGLE/"):
		self.task = task
		self.mask_img = nib.load(mask_path).get_fdata()
		self.filename = os.listdir(HCP_path)
		self.task_file = [name for name in self.filename if task in name]
		if load_num != None:
			self.task_file = self.task_file[:load_num]
		fmri_masked_list = []
		for file in self.task_file:
			fmri_masked_list.append(np.load(HCP_path + file))

		fmri_masked = np.concatenate(fmri_masked_list, axis=0)

		x_train_3D = inverse_transform(fmri_masked, self.mask_img)

		self.img = torch.tensor(x_train_3D, dtype=torch.float)
		del x_train_3D
		self.img = self.img.permute(3, 0, 1, 2)
		self.img = self.img.unsqueeze(1)

	def __getitem__(self, item):

		return self.img[item]

	def __len__(self):
		return self.img.shape[0]

class LoadHCPSub(data.Dataset):
	def __init__(self, task="EMOTION", load_num=None,
				mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
				HCP_path="/home/public/ExperimentData/HCP900/HCP_data/SINGLE/"):
		self.task = task
		self.mask_img = nib.load(mask_path).get_fdata()
		self.filename = os.listdir(HCP_path)
		self.task_file = [name for name in self.filename if task in name]
		self.hcp_path = HCP_path

		if load_num != None:
			self.task_file = self.task_file[:load_num]
		# fmri_masked_list = []
		# for file in self.task_file:
		# 	fmri_masked = np.load(HCP_path + file)
		# 	x_train_3D = inverse_transform(fmri_masked, self.mask_img)
		# 	img = torch.tensor(x_train_3D, dtype=torch.float)
		# 	img = img.permute(3, 0, 1, 2)
		# 	img = img.unsqueeze(0)
		# 	fmri_masked_list.append(img)
		#
		# self.img = torch.cat(fmri_masked_list, dim=0)

	def __getitem__(self, item):
		fmri_masked = np.load(self.hcp_path + self.task_file[item])
		x_train_3D = inverse_transform(fmri_masked, self.mask_img)
		img = torch.tensor(x_train_3D, dtype=torch.float)
		del x_train_3D
		img = img.permute(3, 0, 1, 2)
		return img

	def __len__(self):
		return len(self.task_file)
