import torch
from torch import nn
# aliter: import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
import random

import numpy as np
import cv2
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from torchvision import models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defining model
class FSRCNN(nn.Module):
    def __init__(self, scale_factor = 2, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU()
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU()]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU()])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU()])
        self.mid_part = nn.Sequential(*self.mid_part)
        # In Python, the * operator is used for unpacking an iterable (like a list or tuple) into individual elements.
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

    def forward(self, x):
        x = self.first_part(x)
        if torch.isnan(x).any():
            print("NaN detected in first part")
        x = self.mid_part(x)
        if torch.isnan(x).any():
            print("NaN detected in mid part")
        x = self.last_part(x)
        if torch.isnan(x).any():
            print("NaN detected in last part")
        return x
