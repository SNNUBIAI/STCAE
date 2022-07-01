import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseAttention(nn.Module):
	def __init__(self, in_channels=32):
		super(ChannelWiseAttention, self).__init__()
		self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels//4)
		self.fc2 = nn.Linear(in_features=in_channels//4, out_features=in_channels//4)
		self.fc3 = nn.Linear(in_features=in_channels//4, out_features=in_channels)

	def forward(self, x):
		"""
		:param x: (B, C, D, H, W)
		:return: (B, C, D, H, W)
		"""
		batch, channels, d, h, w = x.size()
		feature = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(batch, channels)
		feature = F.gelu(self.fc1(feature))
		feature = F.gelu(self.fc2(feature))
		feature = self.fc3(feature)
		ca = torch.sigmoid(feature)

		ca_weight = ca.view(batch, channels, 1, 1, 1)
		ca = ca_weight.expand_as(x).clone()
		return ca, ca_weight

class SpatialAttention(nn.Module):
	def __init__(self, in_channels=32):
		super(SpatialAttention, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, padding=3, kernel_size=7)
		self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, padding=1, kernel_size=3)

	def forward(self, x):
		x = F.gelu(self.conv1(x))
		sa_weight = self.conv2(x)
		sa = torch.sigmoid(sa_weight)
		return sa, sa_weight