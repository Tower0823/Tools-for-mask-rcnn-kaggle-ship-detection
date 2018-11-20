# -*- coding: utf-8 -*-
## 将kaggle数据的csv标注转换为带有Mask信息的COCO数据集需要的.json格式
#注意初始的csv文件中，一行是一张图像的一个目标的掩码编码，图像大小:768x768
'kaggle_csv_format:[51834,9,52609,9,……54906,6]'
'COCO_RLE_format:[51833,9,768-9,9,768-9,9,……,768*768-768*4-5*9-51833]'
'''
mask掩码，0:background 1:target
将768*768的二维矩阵先从左到右再从上到下拉成一列，
对应的二维坐标变成一维的顺序索引

kaggle_csv_format:[51834,9,52609,9,……54906,6]
51834%768=378 表示第一个像素点的x坐标，即矩阵中的所在列，
51824//768=69,表示第一个像素点的y坐标，即矩阵中的所在行，
9表示从51834开始往后的9个像素点都是1，从52609往后的9个像素点都是1,……

COCO_RLE_format:[51833,9,768-9,9,768-9,9,……,768*768-768*4-5*9-51833]
分别表示连续的0,1的个数，注意！[]中从左到右奇数位置一定表示0的个数，
比如[0,6,1]为[1,1,1,1,1,1,0];[4,2,2]为[0,0,0,0,1,1,0,0]

但是注意在csv中一张图像的标注可能有很多行
表示一个图像里的框的个数有很多个，一个框一行
在json中也是一个框一个{segmentation}字典
'''
import os
import json
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from skimage import data,segmentation,measure,morphology,color
import pandas as pd
import shutil
import cv2

def mask2K_rle(mask):
    # '初始为1的像素所在的im2col的索引，和连续多少个像素为1'
    # '接着下一个为1的像素的索引，和连续为1的像素个数……直到结束'
    '''
        输入mask,(768,768)矩阵
        返回的是kaggle格式的rle编码与rle编码类似，但是不是coco的rle编码
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '注意！第一步一定要先转置，因为真实的k_rle是先从上到下，即纵向编码的'
    mask=mask.T
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    col_list=mask.reshape(768*768,)
    res_rle=[]
    pos=0
    while pos<768*768:
        len_1=0
        while pos<768*768 and col_list[pos]==0:
            pos+=1
        if pos>=768*768:
            break
        #######################################################
        ##因为！！！在kaggle中图像按列拉成的索引是从1开始的！
        #所以要pos+1
        res_rle.append(pos+1)
        #######################################################
        while col_list[pos]==1:
            len_1+=1
            pos+=1
        res_rle.append(len_1)
    if len(res_rle)%2!=0:
        ipdb.set_trace()
        print("res_rle不是偶数形式！所以去掉最后一个！")
        return res_rle[:-1]
    return res_rle

def mask2poly(mask):
    '''
    输入的是由kaggle中的csv解码得到的0-1二值mask矩阵，
    其中0为背景，1为目标，
    返回的是目标轮廓的顶点坐标按x,y顺序的list
    '''
    poly=[]
    img_tmp=np.array(mask*255,dtype=np.uint8)
    ##########################################################
    # 格式转换！np.uint8，255和浅拷贝!
    img_tmp1=img_tmp.copy()
    _,my_contours,_=cv2.findContours(img_tmp1,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    for node in my_contours[0]:
        poly.append(node[0][0])
        poly.append(node[0][1])
    poly_=[int(a) for a in poly]
    return poly_

def K_rle2mask(mask_rle, shape=(768,768)):
    '''
    mask_rle:原始的csv的mask编码[51834,9,52609,9,……54906,6]
    shape: (height,width)此处所有图像(768,768)
    返回的矩阵是图像的mask 0-1 二值矩阵与原始图像相同大小
    '''
    '一张图像上的一个目标对应一个mask不同目标虽然在一张图像上，但是也画在不同的mask中'
    if type(mask_rle)==float:
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        return img.reshape(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    total_num=0
    for tmp_num in s[1:][::2]:
        total_num=total_num+int(tmp_num)
    ##Kaggle中的rle是按列拉成的，所以要取转置！
    return img.reshape(shape).T


if __name__=='__main__':
    raw_csv='./detectron/datasets/data/kaggle_ship/train_ship_segmentations.csv'
    csv_masks=pd.read_csv(raw_csv)
    raw_img_path='./detectron/datasets/data/kaggle_ship/train/'
    train_img_path='./detectron/datasets/data/kaggle_ship/train2/'#把有目标的图像单独分出来
    new_json='./detectron/datasets/data/kaggle_ship/ship_coco_poly.json'
    mkdir=lambda dir:os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(train_img_path)
    
    '''
        只有一类ship,且info和licenses可以写定，无关紧要
        images是一张图一个的
        annotations是一张图中的一个目标一个，所以annotations比较多
        且annotations中的bbox和segmentations也是一个目标一个的
    '''
    coco_dict={}
    coco_dict["info"]={"description": "Kaggle Ship 2018 Dataset",
                        "url": " ",
                        "version": "1.0",
                        "year": 2018,
                        "contributor": "Kaggle ship data converted by tower",
                        "date_created": "2018/09/07"},
    coco_dict["images"]=[]
    coco_dict["licenses"]=[{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                            "id": 1,
                            "name": "Attribution-NonCommercial-ShareAlike License"}],
    coco_dict["annotations"]=[]
    coco_dict["categories"]=[{"supercategory":"none","id":1,"name":"ship"}]
    
    cur_imgid=20180000000
    cur_annoid=201900000000
    imgname_dict={}
    for csv_mask in csv_masks.itertuples():
        ##csv_mask每一行的属性有ImageId和EncodedPixels
        if type(csv_mask.EncodedPixels)==float:
            continue#跳过没有目标的整张负样本
        img_name=csv_mask.ImageId
        if not img_name in imgname_dict:
            cur_imgid=cur_imgid+1
            imgname_dict[img_name]=cur_imgid
            #若需要得到过滤掉没有目标而分离出有目标的图像，取消注释下一行
            # shutil.copy(raw_img_path+img_name,train_img_path+img_name)
            print("========img_num:%s========="%str(cur_imgid))
        cur_mask=K_rle2mask(csv_mask.EncodedPixels)
        ##这里注意，因为标签是一个目标一行，虽然使用循环，一个mask只有一个连通区域
        for region in measure.regionprops(cur_mask): 
            #得到mask每一个连通区域属性集，region.area可以获取连通区域的面积
            ##最小外接矩形,minr左上角所在行，minc左上角所在列，maxr右下角所在行+1，maxc右下角所在列+1
            '高为(maxr-minr),宽为(maxc-minc),bbox=[x,y,w,h]'
            minr, minc, maxr, maxc = region.bbox
            cur_coco_bbox=[minc,minr,(maxc-minc),(maxr-minr)]
            cur_coco_bbox=[int(a) for a in cur_coco_bbox]
            cur_area=region.area
        cur_img_dict={}
        cur_img_dict["licenses"]=1
        cur_img_dict["file_name"]=img_name
        cur_img_dict["coco_url"]=" "
        ## 此次比赛数据固定大小，因此不读入
        # cur_img_dict["height"]=cur_mask.shape[0]
        # cur_img_dict["width"]=cur_mask.shape[1]
        cur_img_dict["height"]=768
        cur_img_dict["width"]=768
        cur_img_dict["data_captured"]="2018-09-07 11:07:11"
        cur_img_dict["flickr_url"]=" "
        cur_img_dict["id"]=cur_imgid
        coco_dict["images"].append(cur_img_dict)
        #-----------------------------------------------------------------------------------------
        cur_anno_dict={}
        seg_list=mask2poly(cur_mask)
        cur_anno_dict["segmentation"]=[seg_list]
        cur_anno_dict["area"]=cur_area
        '''注意！Detectron中会将iscrowd=1即使用RLE编码的样本过滤掉，所以只能使用多边形标注！
        若iscrowd=0表示mask是多边形的顶点坐标'''
        cur_anno_dict["iscrowd"]=0
        cur_anno_dict["image_id"]=imgname_dict[img_name]
        cur_anno_dict["bbox"]=cur_coco_bbox
        cur_anno_dict["category_id"]=1
        cur_anno_dict["id"]=cur_annoid
        cur_annoid=cur_annoid+1
        print("anno_num: "+str(cur_annoid))
        ##在此处将值写定，因为这次kaggle_ship_competition的图像大小都是768*768的
        coco_dict["annotations"].append(cur_anno_dict)
    ##=================================================================
    res_json=json.JSONEncoder().encode(coco_dict)
    with open(new_json,'w+') as f_new:
        f_new.write(res_json)
        print("成功写入带有Mask信息的COCO格式的instances_train2014.json文件中！！！")
    with open('./detectron/datasets/data/kaggle_ship/backup_json.txt','w+') as f_txt:
        f_txt.write(res_json)
        print("成功备份带有Mask信息的COCO格式的backup_json.txt文件中！！！")