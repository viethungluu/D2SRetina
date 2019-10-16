class LrScheduler(object):
	"""docstring for LrScheduler"""
	def __init__(self, epoch_decay_start, n_epoch, lr):
		super(LrScheduler, self).__init__()
		
		self.epoch_decay_start = epoch_decay_start
		self.n_epoch = n_epoch
		self.lr = lr

		self.mom1 = 0.9
		self.mom2 = 0.1

	def adjust_learning_rate(self, optimizer, epoch, optim_type):
		if epoch < self.epoch_decay_start:
			lr = self.lr
		else:
			lr = float(self.n_epoch - epoch) / (self.n_epoch - self.epoch_decay_start) * self.lr
		
		if epoch < self.epoch_decay_start:
			beta1 = self.mom1
		else:
			beta1 = self.mom2

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
			
			if optim_type == "Adam":
				param_group['betas'] = (beta1, 0.999) # Only change beta1