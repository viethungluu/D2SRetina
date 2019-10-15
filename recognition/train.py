import numpy as np
import argparse, sys

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
from losses import *
from trainer import fit, train_coteaching, eval_coteaching
from scheduler import LrScheduler, adjust_batch_size

from visualization import visualize_images
from contanst import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
# dataset params
parser.add_argument('--coco-path', type=str, help='', default='')
parser.add_argument('--max_image_size', type=int, help='Resize/padding input image to input_size.', default=224)
parser.add_argument('--augment', help='Add data augmentation to training', action='store_true')
parser.add_argument('--num_workers', type=int, help='Number of workers for data loader', default=1)
# model params
parser.add_argument('--backbone', type=str, help='ResNet18/34/50/101/152', default='ResNet50')
parser.add_argument('--batch_sampler', type=str, help='balanced', default = 'balanced')
parser.add_argument('--hard_mining', help='Learn from hard samples instead of easy ones', action='store_true')
parser.add_argument('--optim', help='Optimizer to use: SGD or Adam', default='Adam')
# triplet params
parser.add_argument('--soft_margin', help='Use soft margin.', action='store_true')
# training params
parser.add_argument('--lr', type = float, default=3.0)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--epoch_decay_start', type=int, default=30)
parser.add_argument('--batch_size', type=int, help="Mini-batch size.", default=8)

parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--save_freq', type=int, default=5)
# test/finetuning params
parser.add_argument('--snapshot', type=str, help='Resume training from snapshot', default=None)

args = parser.parse_args()

cuda = torch.cuda.is_available()
# Set up data loaders parameters
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if cuda else {} #

# Seed
torch.manual_seed(args.seed)
if cuda:
	torch.cuda.manual_seed(args.seed)

def plot_embeddings(embeddings, targets, classes, xlim=None, ylim=None):
	plt.figure(figsize=(10, 10))
	for i in range(n_classes):
		inds = np.where(targets==i)[0]
		plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5)
	if xlim:
		plt.xlim(xlim[0], xlim[1])
	if ylim:
		plt.ylim(ylim[0], ylim[1])

	plt.legend(classes)

def extract_embeddings(dataloader, model, embedding_size=2048):
	with torch.no_grad():
		model.eval()
		embeddings = np.zeros((len(dataloader.dataset), embedding_size))
		labels = np.zeros(len(dataloader.dataset))
		k = 0
		for images, target in dataloader:
			if cuda:
				images = images.cuda()
			embeddings[k:k+len(images)] = model.forward(images).data.cpu().numpy()
			labels[k:k+len(images)] = target.numpy()
			k += len(images)
	return embeddings, labels

def model_evaluation(model, train_loader, val_loader, test_loader, plot=True, embedding_size=2048):
	train_embeddings_otl, train_labels_otl 	= extract_embeddings(train_loader, model, embedding_size=embedding_size)
	val_embeddings_otl, val_labels_otl 		= extract_embeddings(val_loader, model, embedding_size=embedding_size)
	test_embeddings_otl, test_labels_otl 	= extract_embeddings(test_loader, model, embedding_size=embedding_size)

	if plot:
		embeddings_otl = np.concatenate((train_embeddings_otl, val_embeddings_otl, test_embeddings_otl))
		embeddings_tsne = TSNE(n_components=2).fit_transform(embeddings_otl)

		labels_otl = np.concatenate((train_labels_otl, val_labels_otl, test_labels_otl)) 

		plot_embeddings(embeddings_tsne, labels_otl, classes)
		plot_embeddings(embeddings_tsne[:train_embeddings_otl.shape[0], ...], train_labels_otl, classes)
		plot_embeddings(embeddings_tsne[train_embeddings_otl.shape[0]: train_embeddings_otl.shape[0] + val_embeddings_otl.shape[0], ...], val_labels_otl, classes)
		plot_embeddings(embeddings_tsne[train_embeddings_otl.shape[0] + val_embeddings_otl.shape[0]:, ...], test_labels_otl, classes)

	clf = KNeighborsClassifier(n_neighbors=6, metric='l2', n_jobs=-1, weights="distance")
	clf.fit(np.concatenate((train_embeddings_otl, val_embeddings_otl)), np.concatenate((train_labels_otl, val_labels_otl)))
	y_pred = clf.predict_proba(test_embeddings_otl)
	return y_pred, test_labels_otl
	
def main():
	transforms_args = [transforms.ToTensor(), transforms.Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0])]

	train_dataset 	= CoCoDataset(args.coco_path, "training", transform=transforms.Compose(transforms_args))
	test_dataset 	= CoCoDataset(args.coco_path, "validation_wo_occlusion", transform=transforms.Compose(transforms_args))
			
	train_loader 	= DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
	test_loader 	= DataLoader(test_dataset, 	batch_size=1, shuffle=False, **kwargs)

	# init model
	model, optim_state_dict = load_model(args.backbone, args.snapshot)
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
	loss_fn = CoTeachingTripletLoss(self_taught=args.self_taught, soft_margin=args.soft_margin, hard_mining=args.hard_mining)
	# define learning rate scheduler
	lr_scheduler = LrScheduler(args.epoch_decay_start, args.n_epoch, args.lr)

	train_log = []
	for epoch in range(1, args.n_epoch + 1):
		lr_scheduler.adjust_learning_rate(optimizer1, epoch - 1, args.optim)

		train_loss_1, train_loss_2, total_train_loss_1, total_train_loss_2 = \
			train_coteaching(train_loader, loss_fn, model1, optimizer1, model2, optimizer2, rate_schedule, epoch, cuda)

		if epoch % args.eval_freq == 0:
			test_loss_1, test_loss_2, test_acc_1, test_acc_2 = \
				eval_coteaching(model1, model2, test_loader, loss_fn, cuda, metric_acc=metric_acc)
			
			train_log.append([train_loss_1, train_loss_2, total_train_loss_1, total_train_loss_2, test_loss_1, test_loss_2])
			print('Epoch [%d/%d], Train loss1: %.4f/%.4f, Train loss2: %.4f/%.4f, Test accuracy1: %.4F, Test accuracy2: %.4f, Test loss1: %.4f, Test loss2: %.4f' 
				% (epoch, args.n_epoch, train_loss_1, total_train_loss_1, train_loss_2, total_train_loss_2, test_acc_1, test_acc_2, test_loss_1, test_loss_2))

			# visualize training log
			train_log_data = np.array(train_log)
			legends = ['train_loss_1', 'train_loss_2', 'total_train_loss_1', 'total_train_loss_2', 'test_loss_1', 'test_loss_2']
			styles = ['b--', 'r--', 'b-.', 'r-.', 'b-', 'r-']
			epoch_count = range(1, train_log_data.shape[0] + 1)
			for i in range(len(legends)):
				plt.loglog(epoch_count, train_log_data[:, i], styles[i])
			plt.legend(legends)
			plt.ylabel('loss')
			plt.xlabel('epochs')
			plt.savefig(os.path.join(MODEL_DIR, '%s_%s_%.2f.png' % (args.dataset, args.loss_fn, args.keep_rate)))
			plt.clf()

		if epoch % args.save_freq == 0:
			torch.save({
						'model_state_dict': model1.state_dict(),
						'optimizer_state_dict': optimizer1.state_dict(),
						'epoch': epoch
						}, os.path.join(MODEL_DIR, '%s_%s_%s_%.2f_1_%d.pth' % (args.dataset, args.backbone, args.loss_fn, args.keep_rate, epoch)))
			
			torch.save({
						'model_state_dict': model2.state_dict(),
						'optimizer_state_dict': optimizer2.state_dict(),
						'epoch': epoch
						}, os.path.join(MODEL_DIR, '%s_%s_%s_%.2f_2_%d.pth' % (args.dataset, args.backbone, args.loss_fn, args.keep_rate, epoch)))

if __name__ == '__main__':
	main()