import matplotlib.pyplot as plt
import pytorch_lightning as pl
from effdet import get_efficientdet_config
from effdet.efficientdet import *
from effdet.efficientdet import _init_weight_alt, _init_weight
from scipy.stats import pearsonr, spearmanr
import cv2
import numpy as np

from image_formatter import ImageFormatter
#import utils


class EdgeMetric(pl.LightningModule):
    def __init__(self, backbone='d0', lr=0.001, agg='mean', unfreeze_backbone=False,
                 reset_backbone=False, test_dataloader=None):
        super().__init__()
        self.save_hyperparameters()

        self.test_dataloader = test_dataloader

        config = get_efficientdet_config(f'tf_efficientdet_{backbone}')
        config.image_size = [64, 64]
        config.num_classes = 1
        config.min_level = 2
        config.num_levels = 6
        pretrained_backbone = not reset_backbone
        alternate_init = False

        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name, features_only=True,
            out_indices=self.config.backbone_indices or (1, 2, 3, 4),
            pretrained=pretrained_backbone, **config.backbone_args)
        if not unfreeze_backbone:
            self.backbone.requires_grad_(False)

        feature_info = get_feature_info(self.backbone)
        for fi in feature_info:
            fi['num_chs'] *= 2
        
        fpn_channels = 72
        self.class_net = nn.Sequential(
            SeparableConv2d(in_channels=fpn_channels, out_channels=fpn_channels, padding='same'),
            SeparableConv2d(in_channels=fpn_channels, out_channels=fpn_channels, padding='same'),
            SeparableConv2d(in_channels=fpn_channels, out_channels=1, padding='same', bias=True,
                            norm_layer=None, act_layer=None),
            nn.Sigmoid()
        )

        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def aggregate(self, x):
        if self.hparams.agg == 'max':
            return torch.amax(x, dim=(-1, -2))
        elif self.hparams.agg == 'mean':
            return torch.mean(x, dim=(-1, -2))
        else:
            raise NotImplementedError(f'Aggregation "{self.hparams.agg}" not implemented')

    def forward(self, img1, img2, edges):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        feat_edges = self.backbone(edges)
        
        stacked = [torch.cat((l, r, e), dim=1) for l, r, e in zip(feat1, feat2, feat_edges)]
        
        pred = self.class_net(stacked[0])
        return self.aggregate(pred)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def loss_func(self, y_pred, y_true_1, y_true_2):       
        loss = F.l1_loss(y_pred, y_true_1 - y_true_2)
        loss = loss.type(torch.cuda.FloatTensor)
        return loss

    def training_step(self, batch, batch_idx):
        img1, img2, edges, score1, score2 = batch
        result = self.forward(img1, img2, edges)
        loss = self.loss_func(result, score1, score2)
        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, edges, score1, score2 = batch
        result = self.forward(img1, img2, edges)
        loss = self.loss_func(result, score1, score2)
        return {'val_loss': loss}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        print(f"| Train_loss: {avg_loss:.3f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)

        for path_list, score in self.test_dataloader:
            predicted_scores = []
            for index, path_to_image in enumerate(path_list):
                image = ImageFormatter.format_image_path(path_to_image)

                same_video = [x for x in ]
                

