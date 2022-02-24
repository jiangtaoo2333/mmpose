# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-12-13 14:18:45
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2022-01-24 10:32:17
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

dirpath = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(dirpath)
# print(dirpath)
# sys.exit()
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result,RMinference_top_down_pose_model)
from mmpose.datasets import DatasetInfo

# print('---------')
dirpath = osp.dirname(osp.abspath(__file__)).replace('\\','/')
dirpath = osp.dirname(dirpath)




class eyeAlignment():
    
    def __init__(self,
                pose_config = '{}/configs/gaze/2d_kpt_sview_rgb_img/topdown_heatmap/dms/res50_gaze_256x256_second_stage.py'.format(dirpath),
                pose_checkpoint = '{}/work_dirs/res50_gaze_256x256_second_stage/best_gaze_loss_epoch_85.pth'.format(dirpath),
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
    
    def alignment(self,img,box,format='xywh'):
        '''
        box:[x,y,w,h]
        '''
        person = {}
        person['bbox'] = box
        person_results = []
        person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = RMinference_top_down_pose_model(
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

eyeAlign = eyeAlignment()

if __name__ == '__main__':

    filename = './demo/images/6.jpg'

    img = cv2.imread(filename,1)

    pose_results, returned_outputs = eyeAlign.alignment(img,[249,153,145,145])

    #pose_results is a list of dict,keys:bbox keypoints gaze
    #return_outputs is list of dict,keys:heatmap 
    # print(type(pose_results))
    # print(type(returned_outputs))
    # print(len(pose_results))
    # print(len(returned_outputs))
    print(pose_results[0].keys())
    for key,value in pose_results[0].items():
        print('key:',key)
        print('value:',value)
    # print(returned_outputs[0].keys())

    # print(pose_results[0]['keypoints'])

    img = handAlign.save(img,
                filename.replace('.jpg','_res.jpg'),
                pose_results,
                returned_outputs)
    print(img.shape)
    cv2.imwrite(filename.replace('.jpg','_res1.jpg'),img)
