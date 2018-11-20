#-*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import ipdb

def K_rle2mask(mask_rle):
    ##输入的mask_rle是一个字符串，'51834 9 52609 9 ……54906 6'
    '''
    mask_rle:原始的csv的mask编码[51834,9,52609,9,……54906,6]
    shape: (height,width)此处所有图像(768,768)
    返回的矩阵是图像的mask 0-1 二值矩阵与原始图像相同大小
    '''
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
    '注意！一定要先转置，因为真实的k_rle是先从上到下按列取，即纵向编码的'
    return img.reshape(shape).T

################################################################################
##自定义将mask矩阵转为kaggle中类rle格式！！
################################################################################
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
    '注意！第一步一定要先转置，因为真实的k_rle是先从上到下，即纵向编码的'
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


#区域坐标必须是一个区域一个np.array,每个区域是list中的一个元素
##x1表示鼠标点击的坐标，水平方向的，但实际上是所在列
'''
mask=np.zeros((768,768))
area1_cord=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
area2_cord=np.array([[],[],[],[]])
'''
##填充的值，如果是三个通道，可以[1,1,1]或[255,255,255]
# mask_write=cv2.fillPoly(mask,[area1_cord,area1_cord],[1])


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
        # if index==0:
        if line=='ImageId,EncodedPixels\n' or line=='ImageId,EncodedPixels\r\n':
            continue
        print(index,len(lines))
        img_name=line.split(',')[0]
        if len(line)<18:
            empty_list.append(img_name)
            # shutil.copy(raw_all_img+img_name,empty_path+img_name)
            continue
        # score=line.split(',')[1]
        rle=line.split(',')[1]
        txt_name=img_name.split('.')[0]+'.txt'
        with open(os.path.join(dir_path,txt_name),'a+') as f_txt:
            f_txt.write(rle)
        
    return empty_list

if __name__=='__main__':
    '''
    此脚本的作用是，解决一个像素被判断为两个目标的情况！
    '''

    #当前结果对应的csv文件
    # raw_csv='./final.csv'
    raw_csv='./v1_11142256_thz.csv'
    ##存放着一张图像一个txt，每行表示一个物体，一行是一个rle
    dir_path='./txt_results/'
    empty_path='./empty_imgs/'#虽然定义了但是没有用
    os.makedirs(dir_path)
    #删除mask重叠以后的csv
    ##注意每次运行此脚本就要删除之前的结果！
    new_csv='./no_overlap_v1_11142256_thz.csv'
    empty_list=csv2txt(raw_csv,dir_path,empty_path)
    ####==============================================================================
    #写入结果csv头
    with open(new_csv,'a+') as f_new:
        f_new.write('ImageId,EncodedPixels\n')
    
    single_txts=os.listdir(dir_path)
    for ind,txt in enumerate(single_txts):
        print(ind,len(single_txts))
        bg_mask=np.zeros((768,768))
        img_name=txt.split('.')[0]+'.jpg'
        with open(dir_path+txt,'r') as f_txt:
            rle_lines=f_txt.readlines()
            
        for rle_line in rle_lines:
            cur_mask=K_rle2mask(rle_line)
            
            #异或，如果背景已经有了，就不要了
            new_mask=((cur_mask>0)&(bg_mask<1))*1
            #或，把新的加入到背景里去
            bg_mask=((cur_mask>0)|(bg_mask>0))*1
            
            if np.sum(new_mask)==0:
                continue
            if np.sum(new_mask)-np.sum(cur_mask)!=0:
                print("有重叠！",img_name)
                # ipdb.set_trace()
            cur_rle=mask2K_rle(new_mask)
            str_rle=[str(i) for i in cur_rle]
            with open(new_csv,'a+') as f_new:
                f_new.write(img_name+','+' '.join(str_rle)+'\n')