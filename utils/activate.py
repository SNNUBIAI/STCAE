import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show
import nibabel as nib
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")

from utils.transform import transform2d, inverse_transform, Masker
from utils.thresholding import thresholding, flip
from model.architecture import STCAE, STCA, MutiHeadSTCA, MutiHeadSTCAE

class FBNActivate:
	def __init__(self, time_step=284,
				 out_map=32,
				 img_path="/home/public/ExperimentData/HCP900/HCP_data/SINGLE/MOTOR_sub_0.npy",
				 mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
				 device='cuda',
				 model_path="./model_save_dir/hcp_motor_1.pth"):
		self.time_step = time_step
		self.out_map = out_map
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

	def plot_net(self, cut_coords=(5, 10, 15, 20, 25, 30, 35, 40), colorbar=True, threshold=False):
		img = self.imgs.to(self.device)
		_, sa, ca = self.attention(img)
		sa = sa.squeeze(0)
		# sa = (sa - sa.flatten(1).min(dim=1)[0].view(self.out_map, 1, 1, 1).expand_as(sa)) / \
		# 	 (sa.flatten(1).max(dim=1)[0].view(self.out_map, 1, 1, 1).expand_as(sa) -
		# 	  sa.flatten(1).min(dim=1)[0].view(self.out_map, 1, 1, 1).expand_as(sa))
		# sa = sa ** 2
		ca = ca.flatten().detach().cpu().numpy()
		if threshold:
			sa = (sa - sa.flatten(1).mean(dim=1).view(self.out_map, 1, 1, 1).expand_as(sa)) / \
				 (sa.flatten(1).std(dim=1).view(self.out_map, 1, 1, 1).expand_as(sa))
			img2d = self.masker.tensor_transform(sa)
			img2d = thresholding(img2d)
			# img2d[img2d < thresholding] = 0
		else:
			sa = (sa - sa.flatten(1).min(dim=1)[0].view(self.out_map, 1, 1, 1).expand_as(sa)) / \
				 (sa.flatten(1).max(dim=1)[0].view(self.out_map, 1, 1, 1).expand_as(sa) -
				  sa.flatten(1).min(dim=1)[0].view(self.out_map, 1, 1, 1).expand_as(sa))
			sa = sa ** 2
			img2d = self.masker.tensor_transform(sa)
		components_img = self.masker.img2NiftImage(img2d)
		plot_prob_atlas(components_img, title='All components', colorbar=True)
		for i, cur_img in enumerate(iter_img(components_img)):
			plot_stat_map(cur_img, display_mode="z", title="index={} weight={:.4f}".format(i, ca[i]),
						  cut_coords=cut_coords, colorbar=colorbar)
			show()

	def get_components(self, threshold=False):
		img = self.imgs.to(self.device)
		_, sa, ca = self.attention(img)
		sa = sa.squeeze(0)
		sa = (sa - sa.flatten(1).mean(dim=1).view(self.out_map, 1, 1, 1).expand_as(sa)) / \
			 (sa.flatten(1).std(dim=1).view(self.out_map, 1, 1, 1).expand_as(sa))
		img2d = self.masker.tensor_transform(sa)
		if threshold:
			img2d = thresholding(img2d)
		else:
			img2d[np.sum(img2d > 0, axis=1) < np.sum(img2d < 0, axis=1), :] *= -1
			img2d = (img2d - img2d.min(axis=1).reshape(-1, 1)) / (img2d.max(axis=1).reshape(-1, 1) - img2d.min(axis=1).reshape(-1, 1))
		return img2d

class STAIndividual(Masker):
	def __init__(self,
				 mask_path="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz",
				 img_path="/home/public/ExperimentData/HCP900/HCP_RestingonMNI/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz",
				 device="cuda",
				 model_path=None,
				 time_step=40,
				 out_map=64,
				 lr=0.0001):
		super(STAIndividual, self).__init__(mask_path=mask_path)
		self.masker = NiftiMasker(mask_img=mask_path,
								  standardize=True,
								  detrend=1,
								  smoothing_fwhm=6.)
		self.masker.fit()
		self.device = device
		self.img_path = img_path
		self.time_step = time_step
		self.stcae = STCAE(time_step=time_step, out_map=out_map)
		self.stca = STCA(time_step=time_step, out_map=out_map)
		if model_path:
			self.stcae.load_state_dict(torch.load(model_path))
			self.stca.load_state_dict(self.stcae.stca.state_dict())
		self.stca.to(self.device)
		self.stcae.to(self.device)
		self.optimizer = Adam(self.stcae.parameters(), lr=lr)
		self.mse_loss = nn.MSELoss()

	def load_img(self):
		if self.img_path.endswith(".npy"):
			fmri_masked = np.load(self.img_path)
		else:
			fmri_masked = self.masker.transform(self.img_path)
		self.imgs = self.inverse_transform2tensor(fmri_masked).unsqueeze(0)
		self.imgs = self.imgs.to(device=self.device)

	def fit(self, epochs=2):
		self.stca.train()
		for epoch in trange(epochs):
			total_loss = 0
			for i in range(self.imgs.shape[1] - self.time_step):
				x = self.imgs[:, i:i + 40, ...]
				y_signals = self.stcae(x)

				loss = self.mse_loss(x, y_signals)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_loss += loss.item()
			print("\nloss :{}".format(total_loss / (self.imgs.shape[1] - self.time_step)))
		self.stca.load_state_dict(self.stcae.stca.state_dict())

	def save_model(self, model_path):
		torch.save(self.stcae.state_dict(), model_path)

	@torch.no_grad()
	def predict(self, index):
		_, sa, ca = self.stca(self.imgs[:, index:index + self.time_step, ...])
		sa = sa.squeeze(0)
		sa = (sa - sa.flatten(1).mean(dim=1).view(64, 1, 1, 1).expand_as(sa)) / \
			 (sa.flatten(1).std(dim=1).view(64, 1, 1, 1).expand_as(sa))
		img2d = self.tensor_transform(sa)
		#img2d[np.sum(img2d > 0, axis=1) < np.sum(img2d < 0, axis=1)] *= -1
		#img2d = (img2d - img2d.min(axis=1).reshape(-1, 1)) / (img2d.max(axis=1).reshape(-1, 1) - img2d.min(axis=1).reshape(-1, 1))
		return img2d

	def eval(self):
		self.stca.eval()

	def plot_net(self, img2d,
				 cut_coords=(5, 10, 15, 20, 25, 30, 35, 40),
				 colorbar=True,
				 threshold=False,
				 annotate=True):
		if threshold:
			img2d[img2d < threshold] = 0
		components_img = self.img2NiftImage(img2d)
		if annotate:
			for i, cur_img in enumerate(iter_img(components_img)):
				plot_stat_map(cur_img, display_mode="z", title="index={}".format(i),
							  cut_coords=cut_coords, colorbar=colorbar, annotate=annotate)
				show()
		else:
			for i, cur_img in enumerate(iter_img(components_img)):
				print(i+1)
				plot_stat_map(cur_img, display_mode="z",
							  cut_coords=cut_coords, colorbar=colorbar, annotate=annotate)
				show()