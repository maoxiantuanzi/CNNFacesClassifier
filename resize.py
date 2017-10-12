#-*- coding: UTF-8 -*-

import cv2
from PIL import Image
from PIL import ImageDraw
import sys,os

def saveFaces(image_name):# 将图片缩放并保存
    # faces = detectFaces(image_name)
    # if faces:
    save_dir = "/home/pengtt/assignment/MSRA-CFW/Sample/faces_resize"
    if not os.path.isdir(save_dir):# 检测目录是否存在，如果不存在那就创建目录
        os.mkdir(save_dir)
    count = 0

    imga = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    # print(image_name)
    new_dir=os.path.join(save_dir,image_name.split('/')[7])
    if not os.path.isdir(new_dir):# 检测目录是否存在，如果不存在那就创建目录
        os.mkdir(new_dir)
    file_name = os.path.join(new_dir,image_name.split('/')[8])
    print(image_name)
    imga = cv2.resize(imga,(64,64))

            # roi = imga[y1:y2,x1:x2]# 这一步是重点，他就像取一个数组一样，把人脸的部分保存出来
            # cv2.imwrite(file_name,cv2.cvtColor(cv2.resize(roi,(92,112)), cv2.COLOR_BGR2GRAY))# 把图片的size重新设定维（92,112），并将图片转换为灰度图像
    cv2.imwrite(file_name,imga)# 把图片的size重新设定维（92,112），并将图片转换为灰度图像

dataDir="/home/pengtt/assignment/MSRA-CFW/Sample/faces_new"
print("ok")
fileList = []
for parent,dirnames,filenames in os.walk(dataDir):
    # print(parent)
    for filename in filenames:
        fileList.append(os.path.join(parent,filename))


for file in fileList:
    # drawFaces(file)
    saveFaces(file)
