import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import ChannelWiseAttention, SpatialAttention
from model.base_module import ConvBlock, ConvDown, ConvUp

class STCA(nn.Module):
	def __init__(self, time_step=284):
		super(STCA, self).__init__()
		self.time_step = time_step
		self.conv_time = nn.Conv3d(in_channels=self.time_step, out_channels=32, kernel_size=7, padding=3)
		self.conv_block = ConvBlock(layer_num=3, kernel_size=5, padding=2)
		self.ca = ChannelWiseAttention(in_channels=32)
		self.sa = SpatialAttention(in_channels=32)

	def forward(self, x):
		x = F.gelu(self.conv_time(x))
		x = self.conv_block(x)
		ca, ca_weight = self.ca(x)
		x = x * ca
		sa, sa_weight = self.sa(x)
		x = x * sa
		return x, sa_weight, ca

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encode_list = nn.ModuleList([ConvDown() for _ in range(3)]) # 32 16 8

	def forward(self, x):
		for layer in self.encode_list:
			x = layer(x)
		return x

class Decoder(nn.Module):
	def __init__(self, time_step=284):
		super(Decoder, self).__init__()
		self.decode_list = nn.ModuleList([ConvUp() for _ in range(3)]) # 16 32 64
		self.conv_time = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=time_step, kernel_size=7, padding=3),
									   nn.GELU(),
									   nn.Conv3d(in_channels=time_step, out_channels=time_step, kernel_size=1, padding=0))

	def forward(self, x):
		for layer in self.decode_list:
			x = layer(x)
		x = self.conv_time(x)
		return x

class STCAE(nn.Module):
	def __init__(self, time_step=284):
		super(STCAE, self).__init__()
		self.time_step = time_step
		self.stca = STCA(time_step=self.time_step)
		self.encoder = Encoder()
		self.decoder = Decoder(time_step=self.time_step)

	def forward(self, x):
		x, _, _ = self.stca(x)
		x = F.pad(x, (8, 9, 3, 3, 7, 8), "constant", 0)
		encode = self.encoder(x)
		decode = self.decoder(encode)
		decode = decode[:, :, 7:-8, 3:-3, 8:-9]
		return decode