'''
========================================================
Centralize dependencies.

: zach wolpe, 18 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''

# machine learning : torch
from torchvision import datasets, models, transforms
from tempfile import TemporaryDirectory
from torchvision import transforms as T
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch import nn
import torchvision
import torch

# machine learning : external
from deepface import DeepFace


# numeric
import pandas as pd
import numpy as np
import cv2

# vis
import plotly.graph_objects as go
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import PIL

# pythonic
import argparse
import warnings
import pickle
import time
import copy
import os


warnings.filterwarnings('ignore')

