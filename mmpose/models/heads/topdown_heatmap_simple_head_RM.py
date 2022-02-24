# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


@HEADS.register_module()
class RMTopdownHeatmapSimpleHead(TopdownHeatmapSimpleHead):

    def __init__(self,
                 frozen_parameters,
                 in_channels,
                 out_channels,
                 **kwargs):
        super().__init__(in_channels,
                        out_channels,
                        **kwargs)

        self.frozen_parameters = frozen_parameters

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['keypoint_loss'] = self.loss(output, target, target_weight)

        return losses

    def _frozen_parameters(self):
        
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

        for param in self.parameters():
            param.requires_grad = False

    def train(self,mode=True):
        super().train(mode)
        if self.frozen_parameters:
            self._frozen_parameters()
