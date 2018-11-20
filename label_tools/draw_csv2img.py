# -*- coding:utf-8 -*-
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import ipdb
import shutil
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from skimage import data,segmentation,measure,morphology,color
from skimage.measure import find_contours

def K_rle2mask(mask_rle):
    #传入的mask_rle是以空格为间隔的一个长字符串,末尾有没有\n都没关系~~因为有split(' ')就去掉啦
    '''
    mask_rle:原始的csv的mask编码[51834,9,52609,9,……54906,6]
    shape: (height,width)此处所有图像(768,768)
    返回的矩阵是图像的mask 0-1 二值矩阵与原始图像相同大小
    '''
    ##注意输入mask_rle是一个以空格为间隔的字符串！
    '一张图像上的一个目标对应一个mask不同目标虽然在一张图像上，但是也画在不同的mask中'
    shape=(768,768)
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
    # return img.reshape(shape)
    '注意！一定要先转置，因为真实的k_rle是先从上到下按列取，即纵向编码的'
    return img.reshape(shape).T

################################################################################
#### ==================加速且正确的！版本！！！！！！====================
def mask2K_rle(mask):
    # '初始为1的像素所在的im2col的索引，和连续多少个像素为1'
    # '接着下一个为1的像素的索引，和连续为1的像素个数……直到结束'
    ##注意第一个数不确定，不知道是第一个像素出现的索引还是索引+1
    #因为索引是从0开始的，但是估计就差一个像素也不会影响很多，计算还是有一定范围的！
    '''
        输入mask,(768,768)矩阵
        返回的是kaggle格式的rle编码与rle编码类似，但是不是coco的rle编码
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask=mask.T
    '注意！第一步一定要先转置，因为真实的k_rle是先从上到下按列取，即纵向编码的'
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
        # res_rle.append(pos)
        while col_list[pos]==1:
            len_1+=1
            pos+=1
        res_rle.append(len_1)
    # print(res_rle)
    if len(res_rle)%2!=0:
        ipdb.set_trace()
        print("res_rle不是偶数形式！所以去掉最后一个！")
        return res_rle[:-1]
    return res_rle


def csv2txt(csv_path,dir_path,empty_path):
    '''
    输入：
        csv_path: 原始csv路径
        dir_path: 需要写入的一张有结果图像对应一个txt的文件夹
        empty_path: 需要写入的为空的图像的文件夹
    返回：
        empty_list: 为空的list
    '''
    ##注意如果重新跑的话要删除之前的结果！
    # 将初始的单个模型的csv结果写入一张图像一个txt，运行多次就把多个模型的结果写在一个txt中！
    ## 返回的是有结果图像的txt的文件夹和没有结果的图像名字的list,并写入没有结果的汇总的csv
    # empty_path='C:/Users/tower/Desktop/Kaggle_Ship/result_true/empty_1.csv'
    # raw_all_img=os.listdir('G:/test_v2/')
    raw_all_img='G:/test_v2/'
    empty_list=[]
    with open(csv_path,'r') as f_raw:
        lines=f_raw.readlines()
    for index,line in enumerate(lines):
        if index==0:
            continue
        print(index,len(lines))
        img_name=line.split(',')[0]
        if len(line)<18:
            empty_list.append(img_name)
            shutil.copy(raw_all_img+img_name,empty_path+img_name)
            continue
        # score=line.split(',')[1]
        rle=line.split(',')[1]
        txt_name=img_name.split('.')[0]+'.txt'
        with open(os.path.join(dir_path,txt_name),'a+') as f_txt:
            #注意！这里第一个元素写入得分，然后逗号，然后是rle编码！
            # f_txt.write(score+','+rle)
            f_txt.write(rle)
        
    return empty_list

def draw_edges(mask_3, mask, color=(255,255,255)):
    '''
    mask_3: img_h,img_w,3
    mask: img_h,img_w
    '''
    color = (0,255,255)
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for ind, verts in enumerate(contours): #遍历每个连通区域的外边界
        Pts = np.zeros((len(verts),1,2),dtype=np.int32)
        # Subtract the padding and flip (y, x) to (x, y)
        Pts[:,0,:] = np.fliplr(verts) - 1
        cv2.polylines(mask_3,[Pts],True,color,1)
    return mask_3
    
    
    
if __name__=='__main__':
    csv_path='./best.csv'
    mkdir=lambda dir:os.makedirs(dir) if not os.path.exists(dir) else None
    empty_path='./empty_imgs/'
    dir_path='./txt_results/'
    many_res_path='./many_res_img/'
    many_raw_path='./many_raw_img/'
    mkdir(empty_path)
    mkdir(dir_path)
    mkdir(many_res_path)
    mkdir(many_raw_path)
    
    ##第一步，把每张图的结果单独分出来！
    empty_list=csv2txt(csv_path,dir_path,empty_path)
    
    raw_all_img='G:/test_v2/'
    # draw_result='./draw_result/'
    draw_result='./draw_results/'
    mkdir(draw_result)
    
    ##第二步，根据每张图的结果画图！
    results=os.listdir(dir_path)
    for ind,txt_result in enumerate(results):
        print(ind,len(results))
        img_name=txt_result.split('.')[0]+'.jpg'
        # ipdb.set_trace()
        with open(dir_path+txt_result ,'r') as f:
            lines=f.readlines()
        print(img_name)
        im=cv2.imread(raw_all_img+img_name)
        img=im.copy()
        
        for line in lines:
            cur_mask=K_rle2mask(line)
            for region in measure.regionprops(cur_mask): 
                #得到mask每一个连通区域属性集，region.area可以获取连通区域的面积
                ##最小外接矩形,minr左上角所在行，minc左上角所在列，maxr右下角所在行+1，maxc右下角所在列+1
                '高为(maxr-minr),宽为(maxc-minc),bbox=[x,y,w,h]'
                minr, minc, maxr, maxc = region.bbox
                cur_coco_bbox=[minc,minr,(maxc-minc),(maxr-minr)]
                cur_coco_bbox=[int(a) for a in cur_coco_bbox]
                img=draw_edges(img,cur_mask)
                cv2.rectangle(img, (minc,minr), (maxc,maxr), (0,0,255), 1)
                cv2.putText(img, 'ship', (minc, minr - 2),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                
                im_mask=img
                cur_mask_3=np.zeros((768,768,3))
                cur_mask_3[:,:,0]=cur_mask*25#b
                cur_mask_3[:,:,1]=cur_mask*250#g
                cur_mask_3[:,:,2]=cur_mask*200#r
                
        cv2.imwrite(draw_result+img_name,im_mask)
        if len(lines)>7:
            cv2.imwrite(many_res_path+img_name,im_mask)
            shutil.copy(raw_all_img+img_name,many_raw_path+img_name)