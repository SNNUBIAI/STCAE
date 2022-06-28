import numpy as np

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