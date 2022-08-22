import numpy as np

def flip(row):
	if np.sum(row > 0) < np.sum(row < 0):
		row *= -1
	return row

def thresholding(array):
	array1 = array

	for idx, row in enumerate(array):
		row = flip(row)
		row[row < 0] = 0
		T = np.amax(row) * 0.3
		row[np.abs(row) < T] = 0

		row = row / np.std(row)
		array1[idx, :] = row
	return array1