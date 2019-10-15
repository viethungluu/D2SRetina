import os
import sys

import numpy as np
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO