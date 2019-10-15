import sys
import os

import numpy as np
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

from image import resize_image

class CoCoDataset(Dataset):
	"""docstring for NoLabelFolder"""
	def __init__(self, data_dir, set_name, target_size=224, transform=None):		
		self.data_dir 		= data_dir
		self.set_name 		= set_name
		self.target_size 	= target_size

		print("Reading dataset from", os.path.join(data_dir, 'annotations', 'D2S_' + set_name + '.json'))
		self.coco 		= COCO(os.path.join(data_dir, 'annotations', 'D2S_' + set_name + '.json'))
		self.image_ids 	= self.coco.getImgIds()
		self._load_classes()
		self._load_annotations()

		print("Number of classes:", self.num_classes())
		print("Number of images:", len(self.image_ids))
		print("Number of image patches:", self.annotations['imgIds'].shape)
		
		self.transform 	= transform

		super(CoCoDataset, self).__init__()

	def _load_classes(self):
		""" Loads the class to label mapping (and inverse) for COCO.
		"""
		# load class names (name -> label)
		categories = self.coco.loadCats(self.coco.getCatIds())
		categories.sort(key=lambda x: x['id'])

		self.classes 				= {}
		self.coco_labels 			= {}
		self.coco_labels_inverse 	= {}
		for c in categories:
			self.coco_labels[len(self.classes)] = c['id']
			self.coco_labels_inverse[c['id']] = len(self.classes)
			self.classes[c['name']] = len(self.classes)
		self.labels = {}
		for key, value in self.classes.items():
			self.labels[value] = key

		# print(self.coco_labels)
		# print(self.coco_labels_inverse)
		# print(self.classes)
		# print(self.labels)

	def _load_annotations(self):
		self.annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'imgIds': np.empty((0,), dtype=np.int)}

		for i, image_id in enumerate(self.image_ids):
			annotations_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)	
			if len(annotations_ids) == 0:
				continue
			
			# parse annotations
			coco_annotations = self.coco.loadAnns(annotations_ids)
			for idx, a in enumerate(coco_annotations):
				# some annotations have basically no width / height, skip them
				if a['bbox'][2] < 1 or a['bbox'][3] < 1:
					continue

				self.annotations['labels'] = np.concatenate([self.annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
				self.annotations['bboxes'] = np.concatenate([self.annotations['bboxes'], [[
						a['bbox'][0],
						a['bbox'][1],
						a['bbox'][0] + a['bbox'][2],
						a['bbox'][1] + a['bbox'][3],
					]]], axis=0)
				self.annotations['imgIds'] = np.concatenate([self.annotations['imgIds'], [i]], axis=0)

	def _load_image(self, index):
		imgId 	= self.annotations['imgIds'][index]
		bbox 	= self.annotations['bboxes'][index]
		label 	= self.annotations['labels'][index]

		image_info 	= self.coco.loadImgs(self.image_ids[imgId])[0]
		path 		= os.path.join(self.data_dir, 'images', image_info['file_name'])
		image 		= np.asarray(Image.open(path).convert('RGB'))

		image 		= image[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), ...]
		# resize image to target_size
		image, _ 	 = resize_image(image, self.target_size)

		return image, label

	def __getitem__(self, index):
		image, label = self._load_image(index)
		
		if self.transform:
			image = self.transform(image)
		
		return image, label

	def __len__(self):
		# 
		return len(self.annotations['imgIds'])

	def all_targets(self):
		return self.annotations['labels']

	def num_classes(self):
		""" Number of classes in the dataset. For COCO this is 80.
		"""
		return len(self.classes)

	def has_label(self, label):
		""" Return True if label is a known label.
		"""
		return label in self.labels

	def has_name(self, name):
		""" Returns True if name is a known class.
		"""
		return name in self.classes

	def name_to_label(self, name):
		""" Map name to label.
		"""
		return self.classes[name]

	def label_to_name(self, label):
		""" Map label to name.
		"""
		return self.labels[label]

	def coco_label_to_label(self, coco_label):
		""" Map COCO label to the label as used in the network.
		COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
		"""
		return self.coco_labels_inverse[coco_label]

	def coco_label_to_name(self, coco_label):
		""" Map COCO label to name.
		"""
		return self.label_to_name(self.coco_label_to_label(coco_label))

	def label_to_coco_label(self, label):
		""" Map label as used by the network to labels as used by COCO.
		"""
		return self.coco_labels[label]

	def image_aspect_ratio(self, index):
		""" Compute the aspect ratio for an image with image_index.
		"""
		bbox 	= self.annotations['bboxes'][index]
		return float(bbox[2] - bbox[0]) / float(bbox[3] - bbox[1])


if __name__ == '__main__':
	parser 	= argparse.ArgumentParser()
	parser.add_argument('--coco-path', type=str, help='', default='')
	parser.add_argument('--set-name', type=str, help='', default='validation_wo_occlusion')
	parser.add_argument('--num-images', help='Number of images to be shown.', type=int, default=9)
	args 	= parser.parse_args()

	ds = CoCoDataset(args.coco_path, args.set_name)
	
	import matplotlib.pyplot as plt
	plt.figure(figsize=(20, 20))
	columns = 3
	num_images = args.num_images if args.num_images < len(ds) else len(ds)
	for i in range(num_images):
		image, label   = ds._load_image(i)
		ax = plt.subplot(num_images // columns + 1, columns, i + 1)
		ax.imshow(image)
		ax.title.set_text(ds.coco_label_to_name(ds.label_to_coco_label(label)))
	
	plt.tight_layout()
	plt.savefig("debug_regcognition.png")