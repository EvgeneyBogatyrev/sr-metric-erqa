import os
import random
from itertools import chain
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import json
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, Subset

from image_formatter import ImageFormatter


class SRDataset(Dataset):

    def __init__(self, root_path, scores_path, transform=None, cases=None, val=False):
        self.root_path = Path(root_path)
        self.scores_path = Path(scores_path)
        with open(self.scores_path, "r") as f:
            self.scores_dict = json.load(f)
        
        video_names = list(os.listdir(self.root_path))
        if cases is not None:
            video_names = [x for x in video_names if x in cases]
        
        self.images = {}
        for video in video_names:
            self.images[video] = {}
            for sr in os.listdir(root_path / video):
                self.images[video][sr] = []
                for image in os.listdir(root_path / video / sr):
                    self.images[video][sr].append(root_path / video / sr / image)
        
        # Transform part (needs revision)
        self.source_transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(value=0),
            A.RandomSizedCrop((8, 32), 64, 64), # TODO: Check
        ],
            additional_targets={
                'image1': 'image'
            }
        )
        
        self.positive_transform = A.Compose([
            A.ShiftScaleRotate(rotate_limit=5, value=0)
        ])

        degradation_transform = [
            A.RandomBrightnessContrast(brightness_by_max=False),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 20)),
                A.GaussianBlur(),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
            ]),
            A.Sharpen(),
            A.ImageCompression()
        ]

        self.degradation_transform = A.Compose(degradation_transform)

        self.transform = transform
        self.val = val

    def __len__(self):
        total = 0
        for video in self.images.keys():
            total += sum([len(x) for x in self.images[video].values()])

    def __getitem__(self, idx):
        # -> (img1, img2, edges, subj1, subj2)
        
        video = None
        sr = None
        frame_index = None

        buffer = 0
        exit = False
        for video_name in self.images.keys():
            for sr_name in self.images[video_name].keys():
                video_len = len(self.images[video_name][sr_name])
                if buffer + video_len < idx:
                    buffer += video_len
                else:
                    video = video_name
                    sr = sr_name
                    frame_index = idx - buffer
                    exit = True
                    break
            if exit:
                break


        assert video is not None, "Dataloader error"
        assert sr is not None, "Dataloader error"
        assert frame_index >= 0, "Dataloader error"

        ref_image_path = self.images[video][sr][frame_index]
        other_sr = random.choice(self.images[video].keys())
        tgt_image_path = self.images[video][other_sr][frame_index]

        ref_image = cv2.imread(ref_image_path)
        tgt_image = cv2.imread(tgt_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)

        # Transform

        # Get subjective scores
        score1 = self.scores_dict[video + "@" + sr]
        score2 = self.scores_dict[video + "@" + other_sr]

        # return data
        return ..., ..., ..., score1, score2