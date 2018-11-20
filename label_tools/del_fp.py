#-*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import ipdb
import matplotlib.pyplot as plt

################################################################################
##自己定义的将mask矩阵转为kaggle中类rle格式！！
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


#区域坐标必须是一个区域一个np.array,每个区域是list中的一个元素
##x1表示鼠标点击的坐标，水平方向的，但实际上是所在列
'''
mask=np.zeros((768,768))
area1_cord=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
area2_cord=np.array([[],[],[],[]])
'''
##填充的值，如果是三个通道，可以[1,1,1]或[255,255,255]
# mask_write=cv2.fillPoly(mask,[area1_cord,area1_cord],[1])


def My_mouse(event,x,y,flags,param):
    # ipdb.set_trace()
    global img,cur_cord
    img2=img.copy()
    if event==cv2.EVENT_LBUTTONDBLCLK:
        #双击左键添加点
        cur_point=(x,y)
        print(cur_point)
        cv2.circle(img, cur_point, 1, (0, 255, 0), 1)
        cur_cord.append([x,y])
    elif event==cv2.EVENT_RBUTTONDBLCLK:
        ##双击右键去掉最新添加的点！
        if len(cur_cord)>0:
            print("删除点:",cur_cord[-1])
            cur_cord.pop()
            print("现有坐标点:",cur_cord)

if __name__=='__main__':

    #读入画了结果的图的文件夹,每张图中有mask
    imgs_dir='./draw_results/'
    #当前画出来的结果对应的csv文件
    raw_csv='./v3_11150459.csv'
    
    
    ##需要删除的mask中的一个点坐标的文件，包含了图像名称和像素所在的列展开索引值
    cord_txt='./remove_cord_fn_mask.txt'
    
    ##删除了误检以后的csv
    new_csv='./remove_v3_11150459.csv'
    
    with open(raw_csv,'r') as f_raw:
        raw_lines=f_raw.readlines()
    
    
    imgs=os.listdir(imgs_dir)
    # print(imgs)
    for img_ind,img_name in enumerate(imgs):
        print("=================================================")
        cur_cord=[]
        print(str(img_ind+1)+'/'+str(len(imgs)),imgs_dir+img_name)
        # ipdb.set_trace()
        img = cv2.imread(imgs_dir+img_name)
        print(img.shape)
        cv2.namedWindow(img_name,0)
        cv2.resizeWindow(img_name, 768, 768)
        cv2.moveWindow(img_name,500,50)
        while 1:
            cv2.imshow(img_name,img)
            cv2.setMouseCallback(img_name,My_mouse)
            #x,y窗口在屏幕的左上
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(img_name)
                break
        print("需要删除的mask中的点为：",cur_cord)
        for pts in cur_cord:
            pixel_index=pts[0]*768+pts[1]
            with open(cord_txt,'a+') as f_cord:
                f_cord.write(img_name+' '+str(pixel_index)+'\n')
    # ####==============================================================================
    with open(cord_txt,'r') as f_cord:
        cord_lines=f_cord.readlines()
    
    with open(raw_csv,'r') as f_raw:
        raw_lines=f_raw.readlines()
    
    for ind,line in enumerate(raw_lines):
        print(ind)
        tag=0
        num_fp=0
        # ipdb.set_trace()
        if line=='ImageId,EncodedPixels\n' or line=='ImageId,EncodedPixels\r\n':
            with open(new_csv,'a+') as f_new:
                f_new.write(line)
            continue
        ##说明是没有结果的
        if line.split(',')[1]=='\n' or line.split(',')[1]=='\r\n':
            with open(new_csv,'a+') as f_new:
                f_new.write(line)
            continue
        img_name=line.split(',')[0]
        rle_str_list=line.split(',')[1].split(' ')
        
        rm_list=[]
        for cord_line in cord_lines:
            if cord_line.split(' ')[0]==img_name:
                rm_pixel=int(cord_line.split(' ')[1])
                rm_list.append(rm_pixel)
        
        for i in range(0,len(rle_str_list),2):
            # print(i,i+1,len(rle_str_list))
            up=int(rle_str_list[i])
            down=up+int(rle_str_list[i+1])
            for rm_pixel in rm_list:
                if up<=rm_pixel and rm_pixel<=down:
                    num_fp+=1
                    print("删除了误检~~~",num_fp)
                    tag=1
                    break
        if tag==0:
            with open(new_csv,'a+') as f_new:
                f_new.write(line)
        