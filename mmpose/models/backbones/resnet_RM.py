# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from torch.nn.modules.batchnorm import _BatchNorm
from .resnet import ResNet

@BACKBONES.register_module()
class RMResNet(ResNet):


    def __init__(self,
                frozen_parameters,
                depth,
                **kwargs):
        super().__init__(depth,**kwargs)

        self.frozen_parameters = frozen_parameters

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


