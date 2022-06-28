import numpy as np

def IoU(n1, n2):
	"""
	:param n1: 1*N
	:param n2: 1*N
	:return: IoU
	"""
	intersect = np.logical_and(n1, n2)
	union = np.logical_or(n1, n2)
	I = np.count_nonzero(intersect)
	U = np.count_nonzero(union)
	return I / U