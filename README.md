# Tools-for-mask-rcnn-kaggle-ship-detection
使用Facebook开源的Detectron，将Mask R-CNN用于Kaggle Airbus Ship Detection比赛
由于整个项目内容太多，这里只放关键的代码，其余详情参见https://github.com/facebookresearch/Detectron
小菜鸡水一波
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeiznob4pj214f0l5wff.jpg)
单模型结果展示：
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeiyhhj62j20lc0lc41v.jpg)
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeiwsb6naj20lc0lcac6.jpg)
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeiw0zkm1j20lc0lc40i.jpg)
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeiwjrj95j20lc0lcgof.jpg)


##./data_convert
- csv2json.py:将kaggle的csv中rle格式的mask标注转换为Detectron使用的coco的json格式
- csv2json_see.py:同样转换且画出图像
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeie2e8ywj20m80gognm.jpg)
## ./label_tools
主要用于修正标注，因为训练集中的有些标注是错误的。。。
- draw_csv2img.py:直接从可以提交的文件（csv）在测试集上画出结果
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxej6qdc9aj20lc0lcn1b.jpg)
- add_label.py:手动标记mask区域，通过输入提示，按顺时针或逆时针依次双击左键确定轮廓点，双击右键取消点，按q完成一个完整目标
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxej0yqvivj20xa0k7jvg.jpg)
- del_fp.py:删除误检，双击某个点，将结果文件中包含这个点的目标的mask所在的那一行结果删除
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeja3acw4j20vv0lwtc3.jpg)
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxeja9iyfzj20ls0mfq3t.jpg)
 使用滚轮放大缩小，放到最大甚至可以看到每个像素3个通道的值！
 ![](http://ww1.sinaimg.cn/large/8ac5d842ly1fxejaj9wgkj20ls0mfdjs.jpg)
- del_overlap.py:删除结果文件中的交叠，即一个像素被判断属于两个物体，将其归为先出现的那个物体

## ./test_code
遍历整个文件夹test_v2测试每张图像，并直接将结果写入csv中
- infer_simple_tower.py:将文件添加至Detectron-master文件夹下./tools/中，运行
- vis_tower.py:将文件添加至./detectron/utils/下
