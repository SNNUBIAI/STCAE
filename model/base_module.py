import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMixer(nn.Module):
	def __init__(self, kernel_size=3, padding=1, groups=32):
		super(ConvMixer, self).__init__()
		self.conv = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32,
											kernel_size=kernel_size, padding=padding, groups=groups),
								  nn.GELU(),
								  nn.BatchNorm3d(32))
		self.mixer = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1),
								   nn.GELU(),
								   nn.BatchNorm3d(32))
	def forward(self, x):
		x_in = x
		x = self.conv(x)
		x = x + x_in
		x = self.mixer(x)
		return x

class ConvBlock(nn.Module):
	def __init__(self, layer_num=3, kernel_size=3, padding=1, groups=1):
		super(ConvBlock, self).__init__()
		self.conv = nn.ModuleList([nn.Conv3d(in_channels=32, out_channels=32,
											 kernel_size=kernel_size, padding=padding, groups=groups)
								   for _ in range(layer_num)])
		self.bn = nn.BatchNorm3d(32)

	def forward(self, x):
		x_in = x
		for layer in self.conv:
			x = F.gelu(layer(x))
		x = x + x_in
		x = self.bn(x)
		return x

class ConvBlockShortCut(nn.Module):
	def __init__(self, in_channels=32, out_channels=32):
		super(ConvBlockShortCut, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
		self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.conv3 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.bn3 = nn.BatchNorm3d(out_channels)

	def forward(self, x):
		x1 = self.conv1(x)
		x1 = self.bn1(x1)

		x2 = self.conv2(x)
		x2 = self.bn2(x2)

		x3 = self.conv3(x)
		x3 = self.bn3(x3)

		if self.in_channels == self.out_channels:
			x = x1 + x2 + x3 + x
		else:
			x = x1 + x2 + x3
		x = F.gelu(x)
		return x

