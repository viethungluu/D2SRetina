import os
import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

class CoCoDataset(Dataset):
	"""docstring for NoLabelFolder"""
	def __init__(self, data_dir, set_name, transform=None):		
		self.data_dir 	= data_dir
		self.set_name 	= set_name
		self.coco 		= OCO(os.path.join(data_dir, 'annotations', 'D2S_' + set_name + '.json'))
		self.image_ids 	= self.coco.getImgIds()
		self._load_classes()
		self._load_annotations()

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
		self.annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
		for image_id in self.image_ids:
			annotations_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
			
			if len(annotations_ids) == 0:
				continue
			
			# parse annotations
			coco_annotations = self.coco.loadAnns(annotations_ids)
			for idx, a in enumerate(coco_annotations):
				# some annotations have basically no width / height, skip them
				if a['bbox'][2] < 1 or a['bbox'][3] < 1:
					continue

				self.annotations['labels'] = np.concatenate([annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
				self.annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
					a['bbox'][0],
					a['bbox'][1],
					a['bbox'][0] + a['bbox'][2],
					a['bbox'][1] + a['bbox'][3],
				]]], axis=0)

	def _load_image(self, index):
		image_info 	= self.coco.loadImgs(self.image_ids[index])[0]
		path 		= os.path.join(self.data_dir, 'images', image_info['file_name'])
		image 		= Image.open(path).convert('RGB')

	def __getitem__(self, index):
		if self.transform:
			image = self.transform(image)
		return image

	def __len__(self):
		return len(self.annotations)

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

	def image_aspect_ratio(self, image_index):
		""" Compute the aspect ratio for an image with image_index.
		"""
		image = self.coco.loadImgs(self.image_ids[image_index])[0]
		return float(image['width']) / float(image['height'])

