import os

import numpy as np
import argparse, sys
import csv

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from models import load_model
from dataset import CoCoDataset

from samplers import TrainBalancedBatchSampler, TestBalancedBatchSampler
from losses import TripletLoss
from triplet_selectors import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector, AllTripletSelector
from scheduler import LrScheduler
from utils import train_epoch, test_epoch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
# dataset params
parser.add_argument('--coco-path', type=str, help='', default='')
parser.add_argument('--target-size', type=int, help='Resize/padding input image to target-size.', default=224)
parser.add_argument('--num-workers', type=int, help='Number of workers for data loader', default=1)
# model params
parser.add_argument('--backbone', type=str, help='ResNet18/34/50/101/152', default='ResNet50')
parser.add_argument('--optim', help='Optimizer to use: SGD or Adam', default='Adam')
# triplet params
parser.add_argument('--soft-margin', 		help='Use soft margin.', action='store_true')
parser.add_argument('--K', 					type=int, 	default=4, help="Number of samples per class for each mini batch. batch_size = K x P")
parser.add_argument('--P', 					type=int, 	default=8, help="Number of classes for each mini batch. batch_size = K x P")
parser.add_argument('--triplet-selector',	type=str, 	default='all', help='Triplet sampling strategy: "all", "hard", "semi", "random". Default: "all"')

# training params
parser.add_argument('--lr', 				type=float, default=1e-3)
parser.add_argument('--n-epoch', 			type=int, 	default=100)
parser.add_argument('--n-batches', 			type=int, 	default=500, help="Number of mini batches for each training epoch. n_batches for testing epoch is fixed to 100")
parser.add_argument('--epoch-decay-start', 	type=int, 	default=30)

parser.add_argument('--eval-freq', type=int, default=5)
parser.add_argument('--save-freq', type=int, default=5)
# test/finetuning params
parser.add_argument('--snapshot', 		type=str, help='Resume training from snapshot', default=None)
parser.add_argument('--snapshot-path', 	type=str, help='Path to save snapshot', default='.')
parser.add_argument('--logger-dir', 	type=str, help='Path to save log', default='.')

args = parser.parse_args()

cuda = torch.cuda.is_available()
# Set up data loaders parameters
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if cuda else {} #

# Seed
torch.manual_seed(args.seed)
if cuda:
	torch.cuda.manual_seed(args.seed)
	
def main():
	transforms_args = [transforms.ToTensor(), transforms.Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0])]

	train_dataset 	= CoCoDataset(args.coco_path, "training", target_size=args.target_size, transform=transforms.Compose(transforms_args))
	test_dataset 	= CoCoDataset(args.coco_path, "validation_wo_occlusion", target_size=args.target_size, transform=transforms.Compose(transforms_args))

	train_batch_sampler = TrainBalancedBatchSampler(torch.from_numpy(np.array(train_dataset.all_targets())),
													K=args.K,
													P=args.P,
													n_batches=args.n_batches)

	test_batch_sampler = TestBalancedBatchSampler(torch.from_numpy(np.array(test_dataset.all_targets())), 
													K=args.K,
													P=args.P,
													n_batches=args.n_batches)
		
			
	train_loader 	= DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
	test_loader 	= DataLoader(test_dataset, 	batch_sampler=test_batch_sampler, **kwargs)

	# init model
	model, optim_state_dict, init_epoch = load_model(args.backbone, args.snapshot)
	if cuda:
		model.cuda()

	# init optimizer
	if args.optim == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
	elif args.optim == "SGD":
		optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)
	else:
		raise ValueError("Optimizer is not supported")
	
	if optim_state_dict is not None:
		optimizer.load_state_dict(optim_state_dict)

	# define loss function
	if args.triplet_selector == "hard":
		selector = HardestNegativeTripletSelector(args.soft_margin)
	elif args.triplet_selector == "semi":
		selector = SemihardNegativeTripletSelector(args.soft_margin)
	elif args.triplet_selector == "random":
		selector = RandomNegativeTripletSelector(args.soft_margin)
	else:
		selector = AllTripletSelector()
	train_loss_fn 	= TripletLoss(selector, soft_margin=args.soft_margin)
	test_loss_fn 	= TripletLoss(AllTripletSelector(), soft_margin=args.soft_margin)

	# define learning rate scheduler
	lr_scheduler = LrScheduler(args.epoch_decay_start, args.n_epoch, args.lr)

	log_file = os.path.join(args.logger_dir, '%s_%s.csv' % (args.backbone, args.triplet_selector)) 
	for epoch in range(init_epoch, init_epoch + args.n_epoch):
		lr_scheduler.adjust_learning_rate(optimizer, epoch - 1, args.optim)

		train_loss = train_epoch(model, train_loader, train_loss_fn, optimizer, cuda)

		if epoch % args.eval_freq == 0:
			test_loss = test_epoch(model, test_loader, test_loss_fn, cuda)
			
			print('Epoch [%d/%d], Train loss: %.4f, Test loss: %.4f' % (epoch, init_epoch + args.n_epoch, train_loss, test_loss))
			log = [epoch, train_loss, test_loss]
			if os.path.isfile(log_file):
				with open(log_file, mode='a', newline='') as csv_f:
					writer = csv.writer(csv_f)
					writer.writerow(log)
			else:
				with open(log_file, mode='w', newline='') as csv_f:
					writer = csv.writer(csv_f)
					# write header
					writer.writerow(["epoch", "train_loss", "test_loss"])
					writer.writerow(log)

		if epoch % args.save_freq == 0:
			torch.save({
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'epoch': epoch
						}, os.path.join(args.snapshot_path, '%s_%s_%d.pth' % (args.backbone, args.triplet_selector, epoch)))

if __name__ == '__main__':
	main()