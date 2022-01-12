# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-12-07 16:26:51
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2022-01-11 14:23:30
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
import random
import ujson

angleToDist = {-50:-110, -40:-77, -30:-52, -20:-34, -10:-16, 0:0,
                50:110, 40:77, 30:52, 20:34, 10:16}

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

def get_box_landmarks_gaze_DMS(jsonFile,Left=True):

    imgFile = jsonFile.replace('.json','.jpg')
    img = cv2.imread(imgFile,1)
    try:
        ih,iw = img.shape[0:2]
    except:
        print('{} have wrong img'.format(jsonFile))
        return False

    try:
        with open(jsonFile, 'r', encoding='UTF-8') as f:
            json_data = ujson.load(f)
    except:
        print('{} have wrong json'.format(jsonFile))
        return False

    landmarks1 = json_data['shapes'][0]['points']
    landmarks2 = json_data['shapes'][1]['points']
    landmarks3 = json_data['shapes'][2]['points']
    landmarks4 = json_data['shapes'][3]['points']
    landmarks5 = json_data['shapes'][4]['points']
    landmarks6 = json_data['shapes'][5]['points']

    # get 3 part landmark.眼皮，虹膜，瞳孔
    if(Left):
        interior_landmarks = np.array((landmarks1),np.float32)
        iris_landmarks = np.array((landmarks2),np.float32)
        iris_center = np.array((landmarks3),np.float32)
    else:
        interior_landmarks = np.array((landmarks4),np.float32)
        iris_landmarks = np.array((landmarks5),np.float32)
        iris_center = np.array((landmarks6),np.float32)

    if(interior_landmarks.shape[0] != 8 or iris_landmarks.shape[0] != 8 or iris_center.shape[0] !=1):
        try:
            os.remove(jsonFile)
            os.remove(jsonFile.replace('.json','.jpg'))
            print('{} have wrong points'.format(jsonFile))
        except:
            pass

    # get gaze
    if(1):
        # look_vec = list(eval(json_data['eye_details']['look_vec']))[0:2]
        dirname = osp.basename(osp.dirname(osp.dirname(jsonFile)))
        yawn = -int(dirname.split('_')[0][1:])
        pitch = -int(dirname.split('_')[1][1:])
        X = angleToDist[yawn]
        Y = angleToDist[pitch]
        Z = 80
        look_vec = [X,Y,Z]
        look_vec = np.array((look_vec))
        # print(look_vec)
        # print(np.linalg.norm(look_vec))
        look_vec = look_vec / np.linalg.norm(look_vec)
        # print(look_vec)

    # get all landmarks
    if(1):
        landmarks = np.concatenate([interior_landmarks,  # 8
                                    iris_landmarks,  # 8
                                    iris_center])  # 17 in total
        pts = landmarks

    # get box
    if(1):
        minx,miny = landmarks.min(axis=0)
        maxx,maxy = landmarks.max(axis=0)
        eye_width = maxx - minx
        eye_height = maxy - miny

        w_ = max(eye_width,eye_height)
        w_ = w_*1.15

        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2

        minx,maxx = int(midx - w_/2),int(midx + w_/2)
        miny,maxy = int(midy - w_/2),int(midy + w_/2)

        minX = np.clip(minx,0,iw-1)
        minY = np.clip(miny,0,ih-1)
        maxX = np.clip(maxx,0,iw-1)
        maxY = np.clip(maxy,0,ih-1)

    return ([minX,minY,maxX,maxY], landmarks, landmarks[-1], look_vec)

def get_box_landmarks_gaze_UnityEyes(jsonFile):

    ih = 480
    iw = 640

    with open(jsonFile, 'r') as f:
        json_data = ujson.load(f)

    def process_coords(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([(x, ih-y, z) for (x, y, z) in coords],np.float32)

    # get 3 part landmark
    if(1):
        interior_landmarks = process_coords(json_data['interior_margin_2d'])
        iris_landmarks = process_coords(json_data['iris_2d'])
        iris_center = iris_landmarks.mean(axis=0).reshape((1,-1))
        # print(iris_center)
        # iris_center[0] = int(iris_center[0])
        # iris_center[1] = int(iris_center[1])

    # get gaze
    if(1):
        look_vec = list(eval(json_data['eye_details']['look_vec']))[0:3]
        look_vec[1] = -look_vec[1]
        look_vec = np.array((look_vec))

    # get all landmarks 眼皮 虹膜 瞳孔
    if(1):
        landmarks = np.concatenate([interior_landmarks[::2, :2],  # 8
                                    iris_landmarks[::4, :2],  # 8
                                    iris_center[:, :2]])  # 17 in total
        pts = landmarks

    # get box
    if(1):
        minx,miny = landmarks.min(axis=0)
        maxx,maxy = landmarks.max(axis=0)
        eye_width = maxx - minx
        eye_height = maxy - miny

        w_ = max(eye_width,eye_height)
        w_ = w_*1.15

        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2

        minx,maxx = int(midx - w_/2),int(midx + w_/2)
        miny,maxy = int(midy - w_/2),int(midy + w_/2)

        minX = np.clip(minx,0,iw-1)
        minY = np.clip(miny,0,ih-1)
        maxX = np.clip(maxx,0,iw-1)
        maxY = np.clip(maxy,0,ih-1)

    return [minX,minY,maxX,maxY], landmarks, landmarks[-1], look_vec

def getlandmarks(landmarks):
    n = landmarks.shape[0]
    score = [1] * n
    score = np.array((score))
    score = score.reshape((n,1))

    realpoints = np.concatenate((landmarks,score),1)
    realpoints = realpoints.reshape((-1))
    realpoints = realpoints.tolist()
    realpoints = list(map(int,realpoints))
    return realpoints

def getbox(box):
    minx,miny,maxx,maxy = box
    return [int(minx),int(miny),int(maxx-minx),int(maxy-miny)]

if __name__ == '__main__':

    srcDir = 'D:/images_check/gazePoints/'
    targetFile = srcDir + 'gazePoints_val.json'

    dataDict = {}
    images = []
    annotations = []
    categories = [{'supercategory': 'person', 'id': 1, 'name': 'face', 'keypoints': [], 'skeleton': []}]

    imgList = getlist(srcDir,'.jpg')
    random.seed(0)
    random.shuffle(imgList)
    N = len(imgList)
    print('{} Total have {} imgs'.format(srcDir,N))

    imgId = 0
    ptsId = 0
    for imgFile in tqdm(imgList[int(0.8*N):]):
        imgFile = imgFile.replace('\\','/')
        jsonFile = osp.splitext(imgFile)[0] + '.json'

        if('UnityEyes' in jsonFile):
            box, landmarks, eye_c, look_vec = get_box_landmarks_gaze_UnityEyes(jsonFile)
        else:
            res = get_box_landmarks_gaze_DMS(jsonFile,True)
            if False == res:
                os.remove(jsonFile)
                os.remove(jsonFile.replace('.json','.jpg'))
                continue
            else:
                box, landmarks, eye_c, look_vec = res

        annotation = {}
        annotation['image_id'] = imgId
        annotation['id'] = ptsId
        annotation['keypoints'] = getlandmarks(landmarks)
        annotation['num_keypoints'] = 17
        annotation['bbox'] = getbox(box)
        annotation['iscrowd'] = 0
        annotation['category_id'] = 1
        annotation['area'] = 10000
        annotations.append(annotation)
        ptsId += 1

        image = {}
        image['id'] = imgId
        file_name = imgFile.split('images_check/gazePoints/',1)[1]
        image['file_name'] = file_name
        height,width = getHeightWidth(imgFile)
        image['height'] = height
        image['width'] = width
        images.append(image)
        imgId += 1

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