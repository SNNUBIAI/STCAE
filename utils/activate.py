import torch
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show
import nibabel as nib

from utils.transform import transform2d, inverse_transform, Masker
from model.architecture import STCAE, STCA

class FBNActivate:
	def __init__(self, time_step=284,
				 out_map=32,
				 img_path="/home/public/ExperimentData/HCP900/HCP_data/SINGLE/MOTOR_sub_0.npy",
				 mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
				 device='cuda',
				 model_path="./model_save_dir/hcp_motor_1.pth"):
		self.time_step = time_step
		model = STCAE(time_step=time_step, out_map=out_map)
		model.load_state_dict(torch.load(model_path))
		self.attention = STCA(time_step=time_step, out_map=out_map)
		self.attention.load_state_dict(model.stca.state_dict())
		self.attention.to(device)
		self.device = device
		self.attention.eval()
		self.masker = Masker(mask_path=mask_path)
		self.imgs = self.masker.inverse_transform2tensor(img_path).unsqueeze(0)

	def setImg(self, img_path):
		self.imgs = self.masker.inverse_transform2tensor(img_path)

	def plot_net(self, cut_coords=(5, 10, 15, 20, 25, 30, 35, 40), colorbar=True):
		img = self.imgs.to(self.device)
		_, sa, ca = self.attention(img)
		sa = sa.squeeze(0)
		sa = (sa - sa.flatten(1).min(dim=1)[0].view(32, 1, 1, 1).expand_as(sa)) / \
			 (sa.flatten(1).max(dim=1)[0].view(32, 1, 1, 1).expand_as(sa) -
			  sa.flatten(1).min(dim=1)[0].view(32, 1, 1, 1).expand_as(sa))
		sa = sa ** 2
		ca = ca.flatten().detach().cpu().numpy()
		img2d = self.masker.tensor_transform(sa)
		components_img = self.masker.img2NiftImage(img2d)
		plot_prob_atlas(components_img, title='All components', colorbar=True)
		for i, cur_img in enumerate(iter_img(components_img)):
			plot_stat_map(cur_img, display_mode="z", title="index={} weight={:.4f}".format(i, ca[i]),
						  cut_coords=cut_coords, colorbar=colorbar)
			show()

	def plot_net_sigmoid(self, cut_coords=(5, 10, 15, 20, 25, 30, 35, 40), colorbar=True):
		img = self.imgs.to(self.device)
		_, sa, ca = self.attention(img)
		sa = sa.squeeze(0)
		sa = torch.sigmoid(sa)
		ca = ca.flatten().detach().cpu().numpy()
		img2d = self.masker.tensor_transform(sa)
		components_img = self.masker.img2NiftImage(img2d)
		plot_prob_atlas(components_img, title='All components', colorbar=True)
		for i, cur_img in enumerate(iter_img(components_img)):
			plot_stat_map(cur_img, display_mode="z", title="index={} weight={:.4f}".format(i, ca[i]),
						  cut_coords=cut_coords, colorbar=colorbar)
			show()
