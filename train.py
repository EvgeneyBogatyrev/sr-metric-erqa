import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import SRDataset
from metric import EdgeMetric

def train_model():
    #Init Datasets
    train_set = SRDataset("/main/mnt/calypso/25e_zim/metric/dataset", \
    "./subjective_scores.json", banned_frames="./banned_frames.json", cases=["beach"])
    val_set = SRDataset("/main/mnt/calypso/25e_zim/metric/dataset", \
    "./subjective_scores.json", banned_frames="./banned_frames.json", cases=["bridge"])

    #Init Dataloaders
    dl_train = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8)
    dl_val = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=8)

    ## Save the model periodically by monitoring a quantity.
    MyModelCheckpoint = ModelCheckpoint(dirpath='runs/pl_segmentation',
                                        filename='{epoch}-{val_loss:.3f}',
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_top_k=1)

    ## Monitor a metric and stop training when it stops improving.
    MyEarlyStopping = EarlyStopping(monitor = "val_loss",
                                    mode = "min",
                                    patience = 5,
                                    verbose = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[MyEarlyStopping, MyModelCheckpoint],
        logger=False,
    )

    model = EdgeMetric(unfreeze_backbone=True)
    model.to(device)

    trainer.fit(model, dl_train, dl_val)

if __name__ == "__main__":
    train_model()

