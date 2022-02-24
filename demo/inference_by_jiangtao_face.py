# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-12-13 14:18:45
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2022-02-23 17:45:36
* @Description  : 
'''
import cv2
import json
import os
import os.path as osp
import sys
import time
import warnings
from argparse import ArgumentParser
from tqdm import tqdm
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


dirpath = osp.dirname(osp.abspath(__file__)).replace('\\','/')
dirpath = osp.dirname(dirpath)


class handAlignment():

    def __init__(self,
                pose_config = '{}/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/dms/res50_dms_256x256.py'.format(dirpath),
                pose_checkpoint = '{}/work_dirs/res50_dms_256x256/best_NME_epoch_42.pth'.format(dirpath),
                device = 'cuda:0'):
        
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.device = device
        print('self.pose_checkpoint:',self.pose_checkpoint)

        self.pose_model = init_pose_model(
                    self.pose_config, self.pose_checkpoint, device=self.device.lower())
        
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)
    
    def alignment(self,img,box):
        '''
        box:[x,y,w,h]
        '''
        person = {}
        person['bbox'] = box
        person_results = []
        person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=True,
            outputs=None)

        return pose_results,returned_outputs

    def save(self, img, outfile, pose_results,returned_outputs):

        img = vis_pose_result(
                self.pose_model,
                img,
                pose_results,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=0.3,
                radius=4,
                thickness=1,
                show=False,
                out_file=outfile)
        return img

handAlign = handAlignment()

if __name__ == '__main__':

    filename = './demo/images/face.jpg'

    img = cv2.imread(filename,1)

    

    # input box is w y w h format
    # pose_results is a list of dict, keys are bbox and keypoints
    # returned_outputs is a list of dict, keys are heatmap

    pose_results, returned_outputs = handAlign.alignment(img,[495,221,400,400])

    print(pose_results[0]['keypoints'])

    img = handAlign.save(img,
                filename.replace('.jpg','_res.jpg'),
                pose_results,
                returned_outputs)
    print(img.shape)
    cv2.imwrite(filename.replace('.jpg','_res1.jpg'),img)
