# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from .csp_darknet import CSPDarknet
from ..builder import BACKBONES


@BACKBONES.register_module()
class RMCSPDarknet(CSPDarknet):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    def __init__(self, 
                 frozen_backbone=False, 
                 arch='P5', 
                 arch_ovewrite=None, 
                 widen_factor=1.0, 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 **kwargs):
        super().__init__(arch=arch, 
                         arch_ovewrite=arch_ovewrite, 
                         widen_factor=widen_factor,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg, 
                         **kwargs)

        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        self.frozen_backbone = frozen_backbone
        self.stem = ConvModule(1,
                               int(arch_setting[0][0] * widen_factor),
                               3,
                               stride=2,
                               padding=1,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)

    def _freeze_backbone(self):
        if self.frozen_backbone:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(RMCSPDarknet, self).train(mode)
        self._freeze_backbone()

    def init_weights(self, pretrained=None):
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')



