import torch
import numpy as np
import nibabel as nib


def transform2d(img, mask_img):
	"""
	:param img:(D, H, W, T)
	:param mask_img:(D, H, W)->bool
	:return: 2d signal (T, n_features)
	"""
	img = np.array(img)
	mask = np.array(mask_img, dtype=np.bool)
	return img[mask].T

def inverse_transform(n_features, mask_img):
	"""
	:param n_features:(T, n_features)
	:param mask_img: (D, H, W)->bool
	:return: (D, H, W, T)
	"""
	mask_img = np.array(mask_img, dtype=np.bool)
	data = np.zeros(mask_img.shape + (n_features.shape[0], ), dtype=n_features.dtype)
	data[mask_img, :] = n_features.T
	return data

class Masker:
	def __init__(self, mask_path):
		self.mask_img = nib.load(mask_path)
		self._affine = self.mask_img.affine
		self.mask_img = self.mask_img.get_fdata()

	def transform(self, img):
		return transform2d(img, self.mask_img)

	def inverse_transform(self, img2d):
		if type(img2d) == str:
			img2d = np.load(img2d)
		return inverse_transform(img2d, self.mask_img)

	def transform2tensor(self, img3d):
		img3d = torch.tensor(img3d, dtype=torch.float)
		img3d = img3d.permute((3, 0, 1, 2))

		return img3d

	def inverse_transform2tensor(self, img2d):
		img3d = self.inverse_transform(img2d)
		img3d = self.transform2tensor(img3d)
		return img3d

	def tensor_transform(self, img):
		img = img.permute((1, 2, 3, 0)).detach().cpu().numpy()
		img2d = self.transform(img)
		return img2d

	@property
	def affine(self):
		return self._affine

	def img2NiftImage(self, img2d):
		img3d = self.inverse_transform(img2d)
		components_img = nib.Nifti1Image(img3d, affine=self.affine)
		return components_img