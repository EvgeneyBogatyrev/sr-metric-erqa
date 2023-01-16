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

import matplotlib.pyplot as plt

from image_formatter import ImageFormatter


class SRDataset(Dataset):

    def __init__(self, root_path, scores_path, banned_frames=None, transform=None, cases=None, val=False):
        self.root_path = Path(root_path)
        self.scores_path = Path(scores_path)
        with open(self.scores_path, "r") as f:
            self.scores_dict = json.load(f)

        self.banned_frames = None
        if banned_frames is not None:
            with open(banned_frames, "r") as f:
                self.banned_frames = json.load(f)
        
        
        video_names = list(os.listdir(self.root_path))
        if cases is not None:
            video_names = [x for x in video_names if x in cases]
        
        self.images = {}
        for video in video_names:
            if video not in self.scores_dict.keys():
                continue
            self.images[video] = {}
            for sr in os.listdir(self.root_path / video):
                if sr not in self.scores_dict[video].keys():
                    continue
                self.images[video][sr] = []
                for image in os.listdir(self.root_path / video / sr):
                    if self.banned_frames is not None and image in self.banned_frames[video]:
                        continue
                    self.images[video][sr].append(self.root_path / video / sr / image)
        
        
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomSizedCrop((100, 270), 270, 480, 1.777), # TODO: Check
            #A.CLAHE(p=0.1)     # Ablation study
        ],
            additional_targets={'other_image': 'image'}
        )
        self.val = val
        self.live = False

    def __len__(self):
        total = 0
        for video in self.images.keys():
            total += sum([len(x) - len(self.banned_frames[video]) for x in self.images[video].values()])
        return total

    def __getitem__(self, idx):
        # -> (img1, img2, edges, subj1, subj2)
        
        video = None
        sr = None
        frame_index = None

        buffer = 0
        exit = False
        for video_name in self.images.keys():
            for sr_name in self.images[video_name].keys():
                video_len = len(self.images[video_name][sr_name]) - len(self.banned_frames[video_name])
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
        other_sr = random.choice(list(self.images[video].keys()))
        tgt_image_path = self.images[video][other_sr][frame_index]


        ref_image = cv2.imread(str(ref_image_path))
        tgt_image = cv2.imread(str(tgt_image_path))
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
    

        # Transform
        transformed = self.transform(image=ref_image, other_image=tgt_image)
        augmented_ref_image = transformed["image"]
        augmented_tgt_image = transformed["other_image"]

        '''
        print(ref_image_path)
        plt.imshow(augmented_ref_image)
        plt.show()
        print(tgt_image_path)
        plt.imshow(augmented_tgt_image)
        plt.show()
        '''

        ref_image_tensor, tgt_image_tensor, edges = \
            ImageFormatter.format_input_images(augmented_ref_image, augmented_tgt_image)

        # Get subjective scores
        score1 = self.scores_dict[video][sr]
        score2 = self.scores_dict[video][other_sr]

        # return data
        return ref_image_tensor.float(), tgt_image_tensor.float(), edges.float(), score1, score2



class LIVEDataset(Dataset):
    def __init__(self, root_path, scores_path, cases=None, val=False):
        self.root_path = Path(root_path)
        self.videos_paths = []
        self.cases = cases

        for video in os.listdir(self.root_path):
            if cases is not None and video not in cases:
                continue
            for case_name in os.listdir(self.root_path / video):
                if case_name[2:4] == "1_":
                    continue
                self.videos_paths.append(self.root_path / video / case_name)
        
        self.scores_path = Path(scores_path)

        seqs_file = self.scores_path / "seqs.txt"
        data_file = self.scores_path / "data.txt"

        with open(seqs_file, "r") as f:
            seqs = list(f.readlines())

        with open(data_file, "r") as f:
            data = list(f.readlines())

        self.scores = {}

        for seq, dt in zip(seqs, data):
            words = dt.split("\t")
            value = float(words[0])
            video = seq[:2]
            if self.root_path / video / seq[:-5] in self.videos_paths:
                self.scores[self.root_path / video / seq[:-5]] = value

        self.val = val
        self.live = True

    def __len__(self):
        return len(self.videos_paths)
    

    def __getitem__(self, index):
        frames = list(os.listdir(self.videos_paths[index]))
        video_path = self.videos_paths[index]
        words = str(video_path).split("/")
        video_name = words[-2]

        same_video = [x for x in self.videos_paths if str(x).split("/")[-2] == video_name]
        print(same_video)

        other_video = random.choice(same_video)

        print(other_video)
        print(os.listdir(other_video))
        return [self.videos_paths[index] / frame for frame in frames], self.scores[self.videos_paths[index]], \
            [other_video / frame for frame in list(os.listdir(other_video))], self.scores[other_video]



class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, scores_file, banned_file=None, val_cases=[], workers=8):
        super().__init__()

        self.data_dir = data_dir
        self.scores_file = scores_file
        self.banned_file = banned_file
        self.train_cases = ['beach', 'bridge', 'camera', 'cars', 'classroom', \
            'constructor', 'grid', 'pig', 'restaurant', 'seesaw', 'statue', 'textbox']
        for cs in val_cases:
            self.test_cases.remove(cs)
        self.val_cases = val_cases
        self.workers = workers

    def setup(self, stage: Optional[str] = None):
        self.train_set = SRDataset(
            self.data_dir, self.scores_file, self.banned_file, cases=self.train_cases
        )
        self.val_set = SRDataset(
            self.data_dir, self.scores_file, self.banned_file, cases=self.val_cases
        )
        '''
        self.train_subset = Subset(SymbolDataset(
            self.data_dir / 'fannet' / 'train',
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros,
            val=True
        ), list(range(64)))
        self.val_set = SymbolDataset(
            self.data_dir / 'fannet' / 'valid',
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros,
            val=True
        )
        self.sr_set = VSRbenchmark(self.data_dir / 'sr-test', choose_frame=50, train_mode=True)
        '''

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=256, shuffle=True, num_workers=self.workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(self.val_set, batch_size=64, num_workers=self.workers),
        ]
