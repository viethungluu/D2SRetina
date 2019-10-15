class LrScheduler(object):
	"""docstring for LrScheduler"""
	def __init__(self, epoch_decay_start, n_epoch, lr):
		super(LrScheduler, self).__init__()
		
		self.epoch_decay_start = epoch_decay_start
		self.n_epoch = n_epoch
		self.lr = lr

		self.mom1 = 0.9
		self.mom2 = 0.1

	def adjust_learning_rate(self, optimizer, epoch, large_batch, optim_type):
		if large_batch:
			if epoch == 30 or epoch == 60 or epoch == 80:
				self.lr = self.lr / 10.0
		else:
			if epoch < self.epoch_decay_start:
				lr = self.lr
			else:
				lr = float(self.n_epoch - epoch) / (self.n_epoch - self.epoch_decay_start) * self.lr
		
		if epoch < self.epoch_decay_start:
			beta1 = self.mom1
		else:
			beta1 = self.mom2

		for param_group in optimizer.param_groups:
			param_group['lr'] = self.lr
			
			if optim_type == "Adam":
				param_group['betas'] = (beta1, 0.999) # Only change beta1

def adjust_batch_size(data_loader, epoch, large_batch):
	if large_batch:
		if epoch < 30:
			data_loader.n_samples = 5
			data_loader.n_batches = 675
		else:
			data_loader.n_samples = 6
			data_loader.n_batches = 375
	else:
		pass