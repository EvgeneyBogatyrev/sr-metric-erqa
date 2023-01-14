import matplotlib.pyplot as plt
import pytorch_lightning as pl
from effdet import get_efficientdet_config
from effdet.efficientdet import *
from effdet.efficientdet import _init_weight_alt, _init_weight
from scipy.stats import pearsonr, spearmanr
import cv2
import numpy as np

#import utils


class EdgeMetric(pl.LightningModule):
    def __init__(self, backbone='d0', lr=0.001, agg='mean', unfreeze_backbone=False,
                 reset_backbone=False):
        super().__init__()
        self.save_hyperparameters()

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
            SeparableConv2d(in_channels=fpn_channels, out_channels=2, padding='same', bias=True,
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

    def loss_func(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        img1, img2, edges, score1, score2 = batch[0], batch[1], batch[2], batch[3], batch[4]
        result = self.forward(img1, img2, edges)
        real_res = torch.tensor([score1, score2])
        real_res = real_res.type(torch.cuda.FloatTensor)
        loss = self.loss_func(result, real_res)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.base_step(batch, batch_idx, 'Val', figures=True)
        elif dataloader_idx == 1:
            name, ref, tgt, subj = batch

            def _run(gt, image):
                return self(gt, image, True)

            res = utils.patch_metric(_run, ref, tgt, self.config.image_size[0], device=self.device)

            return name, res, subj
        elif dataloader_idx == 2:
            self.base_step(batch, batch_idx, 'Train', log=False, figures=True)

    def validation_epoch_end(self, outputs):
        outputs = outputs[1]

        names = []
        heatmaps = []
        subjectives = []

        for name, heatmap, subjective in outputs:
            names.extend(name)
            heatmaps.append(heatmap)
            subjectives.append(subjective.item())

        # Plot vis
        fig, axes = plt.subplots(nrows=2, ncols=2)
        for ax, name, heatmap in zip(axes.flatten(), names, heatmaps):
            ax.imshow(heatmap.cpu().detach(), vmin=0, vmax=1)
            ax.set_title(f'{name}: {heatmap.mean().item():.2f}')

        plt.tight_layout()
        self.logger.experiment.add_figure(f'Val/SR_res', fig, self.current_epoch)

        # Calculate correlation
        scores = self.aggregate(torch.stack(heatmaps)).tolist()
        pearson = pearsonr(subjectives, scores)[0]
        spearman = spearmanr(subjectives, scores)[0]

        self.log('Val/SR_pearson', pearson, add_dataloader_idx=False)
        self.log('Val/SR_spearman', spearman, add_dataloader_idx=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ref, tgt = batch

        def _run(gt, image):
            return self(gt, image, True)

        pred = utils.patch_metric(_run, ref, tgt, self.config.image_size[0], device=self.device)

        return self.aggregate(pred), pred

