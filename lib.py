import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.utils.data as data
import numpy as np

torch.manual_seed(43)
np.random.seed(43)
random.seed(43)