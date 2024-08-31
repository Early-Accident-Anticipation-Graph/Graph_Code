from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os, sys
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.dim_feat = 4096
 
    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output