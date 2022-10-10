from random import sample

def sampling_fmri(img, sample_num=176):
	"""
	:param img: torch.tensor (time step, D, H, W) or (time step, voxels)
	:param sample_num: get uniform sampling from all time step to sample_num
	:return: (sample_num, D, H, W) or (sample_num, voxels)
	"""
	sample_index = sample([i for i in range(img.shape[0])], sample_num)
	sample_index.sort()
	return img[sample_index]

