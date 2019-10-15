import os

import numpy as np
import argparse, sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from models import load_model
from dataset import CoCoDataset
from utils import extract_embeddings

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

cuda = torch.cuda.is_available()

def random_colors(n):
	ret = []
	r = int(np.random.rand() * 256)
	g = int(np.random.rand() * 256)
	b = int(np.random.rand() * 256)
	step = 256 / n
	for i in range(n):
		r += step
		g += step
		b += step
		r = int(r) % 256
		g = int(g) % 256
		b = int(b) % 256
		ret.append((r, g, b)) 
	return ret

def plot_embeddings(embeddings, targets, n_classes, xlim=None, ylim=None):
	colors = random_colors(n_classes)

	import matplotlib.pyplot as plt
	plt.figure(figsize=(10, 10))
	for i in range(n_classes):
		inds = np.where(targets==i)[0]
		plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
	if xlim:
		plt.xlim(xlim[0], xlim[1])
	if ylim:
		plt.ylim(ylim[0], ylim[1])
	
	plt.tight_layout()
	plt.savefig("embeddings.png")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--coco-path', 		type=str, help='', default='')
	parser.add_argument('--set-name', 		type=str, help='', default='')
	parser.add_argument('--target-size', 	type=int, help='Resize/padding input image to target-size.', default=224)
	parser.add_argument('--snapshot', 		type=str, help='', default=None)
	parser.add_argument('--backbone', 		type=str, help='ResNet18/34/50/101/152', default='ResNet50')
	parser.add_argument('--num-workers', 	type=int, help='Number of workers for data loader', default=1)
	args = parser.parse_args()

	# Set up data loaders parameters
	kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if cuda else {} #

	transforms_args = [transforms.ToTensor(), transforms.Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0])]
	dataset 		= CoCoDataset(args.coco_path, args.set_name, target_size=args.target_size, transform=transforms.Compose(transforms_args))
	data_loader 	= DataLoader(dataset, batch_size=8, **kwargs)

	# init model
	model, _, _ = load_model(args.backbone, args.snapshot)
	if cuda:
		model.cuda()

	embeddings, labels 	= extract_embeddings(data_loader, model, cuda=cuda)
	embeddings_tsne 	= TSNE(n_components=2).fit_transform(embeddings)
	plot_embeddings(embeddings_tsne, labels, dataset.num_classes())

if __name__ == '__main__':
	main()
	