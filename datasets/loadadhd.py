from torch.utils import data
import numpy as np

from utils.transform import Masker
from datasets.preprocessing import sampling_fmri

class LoadADHD40Sample(data.Dataset):
	def __init__(self, img_path="/home/public/ExperimentData/ADHD200/adhd/adhd40.npy",
				 mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
				 sample_num=176):
		self.img_path = img_path
		self.mask_path = mask_path
		self.sample_num = sample_num
		fmri_masked = np.load(img_path)
		self.masker = Masker(mask_path=mask_path)
		self.imgs = self.masker.inverse_transform2tensor(fmri_masked)
		self.imgs = self.imgs.unsqueeze(1)

	def __getitem__(self, item):
		imgs = sampling_fmri(self.imgs, sample_num=self.sample_num)
		return imgs

	def __len__(self):
		return self.imgs.shape[0] // self.sample_num
