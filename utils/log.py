from tensorboardX import SummaryWriter
from datetime import datetime

class LogSave:
	def __init__(self, logdir="./logdir/"):
		self.logdir = logdir
		self.TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
		self.writer = SummaryWriter(self.logdir + self.TIMESTAMP)

	def savelog(self, lr=None, loss=None, acc=None, global_step=None):
		if lr != None:
			self.writer.add_scalar("learning rate", lr, global_step=global_step)

		if loss != None:
			self.writer.add_scalar("loss", loss, global_step=global_step)

		if acc != None:
			self.writer.add_scalar("accuracy", acc, global_step=global_step)

	def saveSingleLog(self, name: str, num: float, global_step=None):
		self.writer.add_scalar(name, num, global_step=global_step)