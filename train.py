import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import SRDataset
from metric import EdgeMetric

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset = SRDataset("C:/SR/code/paper_metric_dataset/SR_dataset_subjective/dataset", \
    "./subjective_scores.json", banned_frames="./banned_frames.json")
    train_loader = DataLoader(dataset)

    model = EdgeMetric()
    model.to(device)
    trainer = pl.Trainer(accelerator="gpu")
    trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == "__main__":
    train()

