import os
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torchsummary import summary

from layers import Bottleneck, BasicBlock

model_urls = {
	'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):
	def __init__(self, block, layers):        
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
		self.avgpool = nn.AvgPool2d(14)
		self.dropout = nn.Dropout2d(p=0.5)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
					kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)        

		x = x.view(x.size(0), -1)

		x = self.dropout(x)
		x = F.normalize(x, p=2, dim=1)

		return x

def load_model(backbone, snapshot=None, imagenet_weights=True):
	# create a place-holder model
	if backbone == "ResNet152":
		model = ResNet(Bottleneck, [3, 4, 6, 3])
	elif backbone == "ResNet101":
		model = ResNet(Bottleneck, [3, 4, 6, 3])
	elif backbone == "ResNet50":
		model = ResNet(Bottleneck, [3, 4, 6, 3])
	elif backbone == "ResNet34":
		model = ResNet(BasicBlock, [3, 4, 6, 3])
	elif backbone == "ResNet18":
		model = ResNet(BasicBlock, [2, 2, 2, 2])
	else:
		raise ValueError("backbone must be one of ResNet18, ResNet34, ResNet50, ResNet101, ResNet152")

	optim_state_dict = None
	init_epoch 		 = 0
	if snapshot is not None:
		# continue training
		checkpoint = torch.load(snapshot)
		if type(checkpoint) is dict:
			model_state_dict = checkpoint['model_state_dict']
			optim_state_dict = checkpoint['optimizer_state_dict']
			init_epoch 		 = checkpoint['epoch']
		else:
			model_state_dict = checkpoint
		
		model.load_state_dict(model_state_dict, strict=False)
	else:
		if imagenet_weights:
			# init a new model with ImageNet weights
			model.load_state_dict(model_zoo.load_url(model_urls[backbone]), strict=False)
	
	return model, optim_state_dict, init_epoch

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	debug_model = load_model("ResNet50").to(device)
	summary(debug_model, (3, 224, 224))