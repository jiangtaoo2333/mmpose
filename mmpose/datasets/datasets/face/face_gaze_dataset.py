# Copyright (c) OpenMMLab. All rights reserved.
import json
import numpy as np
import os
import warnings
from collections import OrderedDict
from mmcv import Config

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_nme,
                                                  keypoint_pck_accuracy)
from mmpose.datasets.builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class FaceGazeDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Face300W dataset for top-down face keypoint localization.

    `300 faces In-the-wild challenge: Database and results.
    Image and Vision Computing (IMAVIS) 2019`.

    The dataset loads raw images and apply specified transforms
    to return a dict containing the image tensors and other information.

    The landmark annotations follow the 68 points mark-up. The definition
    can be found in `https://ibug.doc.ic.ac.uk/resources/300-W/`.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/300w.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        self.db = self._get_db()
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                gaze = obj['gaze'][0:-1]
                gaze = np.array((gaze))

                if 'center' in obj and 'scale' in obj:
                    center = np.array(obj['center'])
                    scale = np.array([obj['scale'], obj['scale']]) * 1.25
                else:
                    center, scale = self._xywh2cs(*obj['bbox'][:4], 1.25)

                image_file = os.path.join(self.img_prefix,
                                          self.id2name[img_id])
                gt_db.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'gaze': gaze,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get inter-ocular distance as the normalize factor, measured as the
        Euclidean distance between the outer corners of the eyes.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Return:
            np.ndarray[N, 2]: normalized factor
        """

        interocular = np.linalg.norm(
            gts[:, 5, :] - gts[:, 11, :], axis=1, keepdims=True)
        return np.tile(interocular, [1, 2])

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        gaze_outputs = []
        gaze_gts = []
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gaze_outputs.append(np.array(pred['gaze']))
            gts.append(np.array(item['joints_3d'])[:, :-1])
            gaze_gts.append(np.array(item['gaze']))
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        gaze_outputs = np.array(gaze_outputs)
        gaze_gts = np.array(gaze_gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('PCK', pck))

        if 'PCKh' in metrics:
            _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks, pckh_thr,
                                               threshold_head_box)
            info_str.append(('PCKh', pckh))

        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
                                                 auc_nor)))

        if 'EPE' in metrics:
            info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        if 'NME' in metrics:
            normalize_factor = self._get_normalize_factor(
                gts=gts, box_sizes=box_sizes)
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        # if 'gaze_loss' in metrics:
        gaze_loss = np.linalg.norm(gaze_outputs-gaze_gts,ord=2,axis=1).mean()
        info_str.append(('gaze_loss',gaze_loss))
        return info_str

    def evaluate(self, outputs, res_folder, metric='NME', **kwargs):
        """Evaluate freihand keypoint results. The pose prediction results will
        be saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str]): For example, ['3', '0', '0', 'W', '/',
                    'i', 'b', 'u', 'g', '/', 'i', 'm', 'a', 'g', 'e', '_', '0',
                    '1', '8', '.', 'j', 'p', 'g']
                :output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['NME','gaze_loss']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []
        for output in outputs:
            gaze = output['gaze']
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'gaze':gaze[i].tolist(),
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value
