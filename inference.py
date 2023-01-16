import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from metric import EdgeMetric
from image_formatter import ImageFormatter

def main():
    
    checkpoint_path = "./checkpoint.ckpt"
    model = EdgeMetric.load_from_checkpoint(checkpoint_path)

