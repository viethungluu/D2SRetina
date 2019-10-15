import os

import numpy as np
import argparse, sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import pickle

from models import load_model
from dataset import CoCoDataset
from utils import extract_embeddings

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

cuda = torch.cuda.is_available()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--coco-path', 		type=str, help='', default='')
	parser.add_argument('--train-set-name', type=str, help='', default='training')
	parser.add_argument('--test-set-name', 	type=str, help='', default='validation_wo_occlusion')

	parser.add_argument('--target-size', 	type=int, help='Resize/padding input image to target-size.', default=224)
	parser.add_argument('--snapshot', 		type=str, help='', default=None)
	parser.add_argument('--backbone', 		type=str, help='ResNet18/34/50/101/152', default='ResNet50')
	parser.add_argument('--snapshot-path', 	type=str, help='Path to save snapshot', default='.')
	parser.add_argument('--num-workers', 	type=int, help='Number of workers for data loader', default=1)

	parser.add_argument('--n-neighbors', 	type=int, help='Number of neighbors for KNN classifier', default=1)

	args = parser.parse_args()

	# Set up data loaders parameters
	kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if cuda else {} #

	transforms_args = [transforms.ToTensor(), transforms.Normalize([127.5, 127.5, 127.5], [1.0, 1.0, 1.0])]
	
	train_dataset	= CoCoDataset(args.coco_path, args.train_set_name, target_size=args.target_size, transform=transforms.Compose(transforms_args))
	test_dataset	= CoCoDataset(args.coco_path, args.test_set_name, target_size=args.target_size, transform=transforms.Compose(transforms_args))

	train_loader 	= DataLoader(train_dataset, batch_size=8, **kwargs)
	test_loader 	= DataLoader(test_dataset, batch_size=8, **kwargs)

	# init model
	model, _, _ = load_model(args.backbone, args.snapshot)
	if cuda:
		model.cuda()

	train_embeddings, train_labels 	= extract_embeddings(train_loader, model, cuda=cuda)
	test_embeddings, test_labels 	= extract_embeddings(test_loader, model, cuda=cuda)

	clf = KNeighborsClassifier(n_neighbors=args.n_neighbors, metric='l2', n_jobs=-1, weights="distance")
	clf.fit(train_embeddings, train_labels)
	pickle.dump(clf, open(os.path.join(args.snapshot_path, '%s_%s_%d.pkl' % (args.backbone, args.triplet_selector, epoch)), 'wb'))

	y_prob = clf.predict_proba(test_embeddings)
	y_pred = np.argmax(y_prob, axis=1)
	print(classification_report(test_labels, y_pred))

if __name__ == '__main__':
	main()
	