# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-12-07 16:26:51
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2022-01-10 09:30:26
* @Description  : 
'''
import json
import numpy as np 
import glob
import os
import os.path as osp
import cv2
import sys 
from tqdm import tqdm

def getnpoints(ptsFile):

    with open(ptsFile,'r') as f:
        lines = f.readlines()

    assert int(len(lines) - 1) % 24 == 0

    nFace = int(len(lines) - 1) / 24
    nFace = int(nFace)

    res = []
    for i in range(nFace):
        npoints = lines[24*i+1].strip().split(' ')[-1]
        npoints = int(npoints)
        res.append(npoints)
    return res

def getPts(ptsFile):

    with open(ptsFile,'r') as f:
        lines = f.readlines()


    assert int(len(lines) - 1) % 24 == 0

    nFace = int(len(lines) - 1) / 24

    res = []
    for i in range(int(nFace)):
        begin = 24*i + 3
        end = 24*i + 24

        pts = lines[begin:end]
        pts = [a.strip().split(' ') for a in pts]
        newpts = []
        for singlepoint in pts:
            singleres = []
            for axis in singlepoint:
                axis = int(float(axis))
                singleres.append(axis)
            newpts.append(singleres)

        pts = np.array(pts)
        newpts = np.array(newpts)
        # print(pts)
        res.append(newpts)
    return res

def isValidPts(ptsFile):

    # 检查图片是否能打开
    filename = os.path.splitext(ptsFile)[0]
    imgFile = filename + '.jpg'
    try:
        img = cv2.imread(imgFile,0)
        height, width = img.shape[0:2]
    except:
        print('{} can not open'.format(imgFile))
        return False

    # 检查特征点的质量，超过图像宽高的过滤
    try:
        ptsList = getPts(ptsFile)
        npointsList = getnpoints(ptsFile)
    except:
        print('{} have wrong'.format(ptsFile))
        return False

    assert len(ptsList) == len(npointsList), "error in ptsFile"

    return True

def getlist(dir,extension):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        print('{} have {} files'.format(root,len(files)))
        for dirname in dirs:
            # print(os.path.join(root, name))
            pass
        for name in files:
            filename,ext = os.path.splitext(name)
            if extension == ext:
                list.append(os.path.join(root,name))
    return list

def getHeightWidth(imgFile):
    img = cv2.imread(imgFile,0)
    height,width = img.shape[0:2]
    return height,width

def getjsonpoints(ptsList,npointsList,i):
    pts = ptsList[i]
    points = npointsList[i]
    if points == 21:
        score = [1] * 21
    elif points == 13:
        score = [1] * 13 + [0] * 8
    assert pts.shape == (21,2),'pts have wrong shape {}'.format(pts.shape)

    score = np.array(score).reshape((21,1))
    realpoints = np.concatenate((pts,score),1)
    realpoints = realpoints.reshape((-1))
    realpoints = realpoints.tolist()
    return realpoints

def getjsonbbox(ptsList,i):
    pts = ptsList[i]
    minx,miny = pts.min(axis=0)
    maxx,maxy = pts.max(axis=0)
    h = maxy - miny
    miny -= 0.3*h
    miny = max(miny,0)
    return [int(minx),int(miny),int(maxx-minx),int(maxy-miny)]

if __name__ == '__main__':

    srcDir = '/jiangtao2/dataset/train/alignment/test'
    targetFile = srcDir + 'face_landmarks_dms_test.json'

    dataDict = {}
    images = []
    annotations = []
    categories = [{'supercategory': 'person', 'id': 1, 'name': 'face', 'keypoints': [], 'skeleton': []}]

    print('srcDir:',srcDir)
    imgList = getlist(srcDir,'.jpg')
    print('Total have {} imgs'.format(len(imgList)))

    imgId = 0
    ptsId = 0
    for imgFile in tqdm(imgList):

        ptsFile = osp.splitext(imgFile)[0] + '.pts'

        # 检查错误样本
        if not os.path.exists(ptsFile):
            print('{} dont have ptsFile'.format(imgFile))
            continue

        if not isValidPts(ptsFile):
            print('{} is not valid'.format(imgFile))
            continue

        ptsList = getPts(ptsFile)
        npointsList = getnpoints(ptsFile)
        for i in range(len(ptsList)):
            annotation = {}
            annotation['image_id'] = imgId
            annotation['id'] = ptsId
            annotation['keypoints'] = getjsonpoints(ptsList,npointsList,i)
            annotation['num_keypoints'] = 21
            annotation['bbox'] = getjsonbbox(ptsList,i)
            annotation['iscrowd'] = 0
            annotation['category_id'] = 1
            annotation['area'] = 10000
            annotations.append(annotation)
            ptsId += 1

            image = {}
            image['id'] = imgId
            file_name = imgFile.split('train/alignment/test/',1)[1]
            image['file_name'] = file_name
            height,width = getHeightWidth(imgFile)
            image['height'] = height
            image['width'] = width
            images.append(image)
            imgId+=1

    dataDict['images'] = images
    dataDict['annotations'] = annotations
    dataDict['categories'] = categories

    # print(type(dataDict))
    # print(dataDict.keys())

    for key,value in dataDict['images'][0].items():
        print('-'*25)
        print(key)
        print(type(value))
        print(value)
    for key,value in dataDict['annotations'][0].items():
        print('-'*25)
        print(key)
        print(type(value))
        print(value)
    for key,value in dataDict['categories'][0].items():
        print('-'*25)
        print(key)
        print(type(value))
        print(value)
    # sys.exit()
    
    print(len(dataDict['images']))
    print(len(dataDict['annotations']))

    with open(targetFile, "w") as f:
        f.write(json.dumps(dataDict, indent=4))


    # jsonFile = srcDir + 'face_landmarks_dms_test.json'
    # with open(jsonFile,'r') as f:
    #     data = json.load(f)
    # print(len(data['images']))
    # print(len(data['annotations']))