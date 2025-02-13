# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import json
from xtcocotools.coco import COCO
from tqdm import tqdm
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

def getlist(dir,extension,Random=False):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)

            if extension == ext:
                list.append(os.path.join(root,name))
                list[-1]  = list[-1].replace('\\','/')
    if Random:
        random.shuffle(list)
    return list

def getbox(json_file):
    with open(json_file,'rb') as f:
        data = json.load(f)
    points = data['shapes'][0]['points']
    x,y,w,h = points[0],points[1],points[2]-points[0],points[3]-points[1]
    return [x,y,w,h]

def main():

    if(1):
        """Visualize the demo images.

        Require the json_file containing boxes.
        """
        parser = ArgumentParser()
        parser.add_argument('pose_config', help='Config file for detection')
        parser.add_argument('pose_checkpoint', help='Checkpoint file')
        parser.add_argument('--img-root', type=str, default='', help='Image root')
        parser.add_argument(
            '--json-file',
            type=str,
            default='',
            help='Json file containing image info.')
        parser.add_argument(
            '--show',
            action='store_true',
            default=False,
            help='whether to show img')
        parser.add_argument(
            '--out-img-root',
            type=str,
            default='',
            help='Root of the output img file. '
            'Default not saving the visualization images.')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
        parser.add_argument(
            '--radius',
            type=int,
            default=4,
            help='Keypoint radius for visualization')
        parser.add_argument(
            '--thickness',
            type=int,
            default=1,
            help='Link thickness for visualization')

        args = parser.parse_args()

        assert args.show or (args.out_img_root != '')

    # coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # img_keys = list(coco.imgs.keys())
    imgList = getlist(args.img_root,'.jpg')

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # process each image
    for i in tqdm(range(len(imgList))):
        print('i:',i)
        print(imgList[i])
        # get bounding box annotations

        image_name = imgList[i]


        # make person bounding boxes
        person_results = []
        person = {}
        person['bbox'] = getbox(image_name.replace('.jpg','.json'))
        person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        pose = pose_results[0]['keypoints']
        with open(image_name.replace('.jpg','.txt'),'w') as f:
            for i in range(pose.shape[0]):
                f.write(str(pose[i][0]) + ' ')
                f.write(str(pose[i][1]) + '\n')
        continue
        # print(type(pose_results))
        # print(pose_results)
        # import time
        # time.sleep(10000)


if __name__ == '__main__':
    main()
