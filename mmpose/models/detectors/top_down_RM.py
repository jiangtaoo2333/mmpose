# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16
from .top_down import TopDown

@POSENETS.register_module()
class RMTopDown(TopDown):

    def __init__(self,
                backbone,
                gaze_head,
                loss_cfg,
                **kwargs):
        super().__init__(backbone,**kwargs)

        self.gaze_head = builder.build_head(gaze_head)
        self.loss_cfg = loss_cfg

    @property
    def with_gaze(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'gaze_head')
    
    def init_weights(self,pretrained=None):
        super().init_weights(pretrained)

        if self.with_gaze:
            self.gaze_head.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                gaze=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths
                  and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, gaze, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, gaze, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output_keypoints = self.keypoint_head(output)
        if self.with_gaze:
            output_gaze = self.gaze_head(output)

        # if return loss mse_loss acc_pose
        losses = dict()
        if self.loss_cfg.keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output_keypoints, target, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output_keypoints, target, target_weight)
            losses.update(keypoint_accuracy)
        if self.loss_cfg.gaze:
            gaze_losses = self.gaze_head.get_loss(
                output_gaze, gaze)
            losses.update(gaze_losses)
        # print('losses',losses)
        # import sys
        # sys.exit()
        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)
        if self.with_gaze:
            gaze = self.gaze_head.inference_model(features)
            result['gaze'] = gaze

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap


        return result