# -*- coding:utf-8-*-

import cv2
from PIL import Image
from PIL import ImageDraw
import sys,os
def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier('/home/pengtt/opencv-2.4.13.3/data/haarcascades/haarcascade_frontalface_alt.xml')# 加载级联分类器，这里使用的是intel训练出来的人脸识别分类器
    # print(img.ndim)
    # print('ok')
    if img.ndim == 3:# 判断图片是否是灰度图像，如果img.ndim==3那就表示不是灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 将图片转化为灰度图像
        # print(gray.ndim)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)# 核心操作，返回图片中所有的人脸的坐标和宽高度
    result = []
    for (x, y, width, height) in faces:
        # print (x, y, width, height)
        result.append((x, y, x+width, y+height))# 将原始数据，转化为人脸的四个点的坐标
    return result
def drawFaces(image_name):
    faces = detectFaces(image_name)# 此处的返回值是一个元组，(x, y, width, height)，每一个元组都包括人脸的（x,y）坐标，还有人脸的宽度和高度，有这些数据，我们就可以把人脸标记出来，比如画一个矩形框出来
    # print faces
    # if faces:
    #     imga = cv2.imread(image_name)# 读取图片
    #     draw_rects(imga, faces, (0,255,0))# 画矩形标记
    #     cv2.imshow('img', imga)# 显示修改后的矩形
    #     cv2.waitKey(0)
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)# 调用cv2的矩形函数，画矩形
def saveFaces(image_name):# 将人脸截取并保存成图片
    faces = detectFaces(image_name)
    if faces:
        save_dir = "/home/pengtt/assignment/MSRA-CFW/Sample/faces_new"
        # print(save_dir)
        if not os.path.isdir(save_dir):# 检测目录是否存在，如果不存在那就创建目录
            os.mkdir(save_dir)
        count = 0
        for (x1, y1, x2, y2) in faces:
            imga = cv2.imread(image_name)
            new_dir=os.path.join(save_dir,image_name.split('/')[7])
            if not os.path.isdir(new_dir):# 检测目录是否存在，如果不存在那就创建目录
                os.mkdir(new_dir)
            file_name = os.path.join(new_dir,image_name.split('/')[8])
            # # print(image_name.split('/')[8])
            roi = imga[y1:y2,x1:x2]# 这一步是重点，他就像取一个数组一样，把人脸的部分保存出来
            cv2.imwrite(file_name,cv2.cvtColor(cv2.resize(roi,(92,112)), cv2.COLOR_BGR2GRAY))# 把图片的size重新设定维（92,112），并将图片转换为灰度图像
            cv2.imwrite(file_name,roi)# 把图片的size重新设定维（92,112），并将图片转换为灰度图像
            # cv2.imshow('aaa',imga)
# drawFaces('E:/0001_00_00_01_0.jpg')
# saveFaces('E:/0001_00_00_01_0.jpg')

dataDir="/home/pengtt/assignment/MSRA-CFW/Sample/thumbnails_features_deduped_sample2"
fileList = []

for parent,dirnames,filenames in os.walk(dataDir):
    for filename in filenames:
        if filename.endswith('jpg'):
            fileList.append(os.path.join(parent,filename))
        # print(parent)

for file in fileList:
    # drawFaces(file)
    saveFaces(file)
