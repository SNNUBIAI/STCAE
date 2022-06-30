import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import Adam, SGD
from tqdm import trange
import argparse

from model.architecture import STCAE
from utils.log import LogSave
from datasets.loadhcp import LoadHCPSub

class Trainer:
	def __init__(self, args):
		self.args = args
		self.model = None
		if self.args.model == 'stcae':
			self.model = STCAE(time_step=args.time_step)
		else:
			print("No such model architecture.")
			exit(0)
		self.model.to(self.args.device)
		if args.load_model:
			self.model.load_state_dict(torch.load("{}{}_{}.pth".format(args.model_path,
																	   args.encoder,
																	   args.load_epochs)))
		if args.parallel:
			self.model = nn.DataParallel(self.model, device_ids=[0, 1])

		self.mse_loss = nn.MSELoss()
		if args.optim == 'sgd':
			self.optimizer = SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
		elif args.optim == 'adam':
			self.optimizer = Adam(self.model.parameters(), lr=args.lr)

		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
														 step_size=args.step_size,
														 gamma=args.gamma)
		print("loading data......")
		self.data_loader = data.DataLoader(LoadHCPSub(task=args.task,
													  load_num=args.load_num,
													  HCP_path=args.img_path,
													  mask_path=args.mask_path),
										   batch_size=args.batch_size,
										   shuffle=True)
		print("complete.")
		self.log = LogSave(logdir=args.logdir)


	def fit(self):
		for epoch in trange(1, self.args.epochs + 1):
			self.train_one_epoch(epoch)

	def train_one_epoch(self, epoch):
		total_loss = 0
		for x_signals in self.data_loader:
			x_signals = x_signals.to(self.args.device)
			y_signals = self.model(x_signals)

			loss = self.mse_loss(x_signals, y_signals)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			total_loss += loss.item()
		self.log.savelog(lr=self.optimizer.state_dict()['param_groups'][0]['lr'],
						 loss=total_loss / len(self.data_loader),
						 global_step=epoch)
		self.scheduler.step()
		self.saveModel(epoch)

	def saveModel(self, epoch):
		model_path = self.args.model_path + "{}_{}.pth".format(self.args.encoder,
															   self.args.load_epochs + epoch)
		if self.args.parallel:
			torch.save(self.model.module.state_dict(), model_path)
		else:
			torch.save(self.model.state_dict(), model_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--device', default='cuda', type=str)
	parser.add_argument('--epochs', default=20, type=int)
	parser.add_argument('--img_path', default="/home/public/ExperimentData/HCP900/HCP_data/SINGLE/", type=str)
	parser.add_argument('--mask_path', default="/home/public/ExperimentData/HCP900/HCP_data/mask_152_4mm.nii.gz", default=str)
	parser.add_argument('--model_path', default="./model_save_dir/", type=str)
	parser.add_argument('--load_model', default=False, type=bool)
	parser.add_argument('--load_epochs', default=0, type=int)
	parser.add_argument('--encoder', default='hcp_motor', type=str)
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--optim', default='adam', type=str)
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--step_size', default=2, type=int)
	parser.add_argument('--gamma', default=0.9, type=float)
	parser.add_argument('--parallel', default=False, type=bool)
	parser.add_argument('--model', default='stcae', type=str)
	parser.add_argument('--time_step', default=284, type=int)
	parser.add_argument('--logdir', default='./logdir/', type=str)
	parser.add_argument('--task', default='MOTOR', type=str)
	parser.add_argument('--load_num', default=40, type=int)
	args = parser.parse_args()

	trainer = Trainer(args=args)
	trainer.fit()
	