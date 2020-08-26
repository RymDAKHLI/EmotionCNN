import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from constants import CLASSES




def resnet18():
    net = torchvision.models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, 7, 2, padding=3, padding_mode='zeros', bias=False)
    net.fc = nn.Linear(in_features=512, out_features=len(CLASSES), bias=True)
    return net
