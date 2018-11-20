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
        # cv2.circle(img2, cur_point, 2, (0, 255, 0), 1)
        cv2.circle(img, cur_point, 1, (0, 255, 0), 1)
        # cv2.point(img, cur_point)
        if len(cur_cord)>0:
            # print(cur_cord[-1][0],cur_cord[-1][1])
            cv2.line(img, (cur_cord[-1][0],cur_cord[-1][1]), cur_point, (0, 255, 0), thickness=1)
        
        cur_cord.append([x,y])
        # print(cur_cord)
    elif event==cv2.EVENT_RBUTTONDBLCLK:
        ##双击右键去掉最新添加的点！
        if len(cur_cord)>0:
            print("删除点:",cur_cord[-1])
            cur_cord.pop()
            print("现有坐标点:",cur_cord)

if __name__=='__main__':
    ##注意！！！cv2不能读入中文路径啊！！！！
    imgs_dir='C:/Users/tower/Desktop/FUSION/final/0.725-v1/replace_all_tower_v1/'
    imgs=os.listdir(imgs_dir)
    
    add_result='C:/Users/tower/Desktop/FUSION/final/0.725-v1/replace_all_tower_final.csv'
    
    ###############################################
    '注意，后面改成a+了，每次运行时要删掉之前的结果！'
    # f_add=open(add_result,'w+')
    ###############################################
    
    print(imgs)
    # ipdb.set_trace()
    # imgs=['C:/Users/tower/Desktop/FUSION/ff3294cb9.jpg']
    for img_ind,img_name in enumerate(imgs):
        ##记录一下已经有的，不能重叠
        bg_mask=np.zeros((768,768))
        print("=================================================")
        print(str(img_ind+1)+'/'+str(len(imgs)),imgs_dir+img_name)
        # ipdb.set_trace()
        img = cv2.imread(imgs_dir+img_name)
        print(img.shape)
        cv2.namedWindow("raw_img",0)
        cv2.resizeWindow("raw_img", 768, 768)
        cv2.moveWindow("raw_img",500,50)
        while 1:
            cv2.imshow('raw_img',img)
            #x,y窗口在屏幕的左上
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('raw_img')
                break
        while 1:
            max_obj=input("按q结束此张图像或输入继续增加的目标个数: ")
            print(max_obj)
            print(type(max_obj))
            if str.isdigit(max_obj):
                max_obj=int(max_obj)
                # break
            else:
                if max_obj=='q':
                    break
                else:
                    print("请输入数字！")
                    continue
        
            for cur_obj in range(max_obj):
                cur_cord=[]
                
                cv2.namedWindow(img_name,0)
                cv2.resizeWindow(img_name, 768, 768)
                cv2.moveWindow(img_name,500,50)
                # cv2.namedWindow(img_name)
                # cv2.namedWindow(img_name,0)
                cv2.setMouseCallback(img_name,My_mouse)
                print("~~~~增加目标："+str(cur_obj+1)+'/'+str(max_obj)+' ~~~~')
                while (1):
                    cv2.imshow(img_name,img)
                    # if input("退出按q") == 'q':#按=键退出
                    if cv2.waitKey(1)&0xFF == ord('q'):#按q键退出
                        cv2.line(img, (cur_cord[0][0],cur_cord[0][1]), (cur_cord[-1][0],cur_cord[-1][1]), (0, 255, 0), thickness=1)
                        break
                # ipdb.set_trace()
                print(cur_cord)
                #得到np.array格式的坐标
                cur_cord=np.array(cur_cord)
                cur_mask=np.zeros((768,768))
                ##要按顺时针才行的！传入的是np.array的list！！！
                # cur_mask=cv2.fillPoly(cur_mask,[cur_cord],[1])
                ##还是要顺时针找凸包，传入的是np.array的坐标
                cur_mask=cv2.fillConvexPoly(cur_mask,cur_cord,1)
                
                #异或，如果背景已经有了，就不要了
                new_mask=((cur_mask>0)&(bg_mask<1))*1
                #或，把新的加入到背景里去
                bg_mask=((cur_mask>0)|(bg_mask>0))*1
                
                cur_rle=mask2K_rle(new_mask)
                print(cur_rle)
                str_krle=[str(num) for num in cur_rle]
                cv2.namedWindow('mask',0)
                cv2.resizeWindow(img_name, 400, 400)
                cv2.moveWindow("mask",800,50)
                while 1:
                    cv2.imshow('mask',cur_mask)
                    if cv2.waitKey(1)&0xFF == ord('q'):#按q键退出
                        cv2.destroyWindow('mask')
                        break
                cv2.destroyAllWindows()
                print(img_name+','+' '.join(str_krle))
                with open(add_result,'a+') as f_add:
                    f_add.write(img_name+','+' '.join(str_krle)+'\n')
