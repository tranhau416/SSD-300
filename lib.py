import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.utils.data as data
import numpy as np
from matplotlib import pyplot as plt

torch.manual_seed(1313)
np.random.seed(1313)
random.seed(1313)