import cv2
from PIL import Image
import numpy as np
src = Image.open('C:/Users/blue_/Desktop/Satellite-Segmentation-master/unet/predict/src.png')
pred = Image.open('C:/Users/blue_/Desktop/Satellite-Segmentation-master/unet/predict/pre1.png')

width = src.size[0]#长度
height = pred.size[1]#宽度unet
tp_1=tn_1=fp_1=fn_1=tp_2=tn_2=fp_2=fn_2=tp_3=tn_3=fp_3=fn_3=tp_4=tn_4=fp_4=fn_4=tp_0=tn_0=fp_0=fn_0=count=0
ka = np.zeros((5,5))
for i in range(5,width):#遍历所有长度的点
    for j in range(0,height):#遍历所有宽度的点
        src_data = (src.getpixel((i, j)))
        pred_data = (pred.getpixel((i,j)))
        ka[src_data][pred_data] += 1
        count += 1
        if src_data == 1 and pred_data == 1:
            tp_1 += 1
        if src_data == 1 and pred_data != 1:
            fn_1 += 1
        if src_data != 1 and pred_data == 1:
            fp_1 += 1
        if src_data != 1 and pred_data != 1:
            tn_1 += 1
        if src_data == 2 and pred_data == 2:
            tp_2 += 1
        if src_data == 2 and pred_data != 2:
            fn_2 += 1
        if src_data != 2 and pred_data == 2:
            fp_2 += 1
        if src_data != 2 and pred_data != 2:
            tn_2 += 1
        if src_data == 3 and pred_data == 3:
            tp_3 += 1
        if src_data == 3 and pred_data != 3:
            fn_3 += 1
        if src_data != 3 and pred_data == 3:
            fp_3 += 1
        if src_data != 3 and pred_data != 3:
            tn_3 += 1
        if src_data == 4 and pred_data == 4:
            tp_4 += 1
        if src_data == 4 and pred_data != 4:
            fn_4 += 1
        if src_data != 4 and pred_data == 4:
            fp_4 += 1
        if src_data != 4 and pred_data != 4:
            tn_4 += 1
        if src_data == 0 and pred_data == 0:
            tp_0 += 1
        if src_data == 0 and pred_data != 0:
            fn_0 += 1
        if src_data != 0 and pred_data == 0:
            fp_0 += 1
        if src_data != 0 and pred_data != 0:
            tn_0 += 1


iou_1 = tp_1/(tp_1+fn_1+fp_1)
iou_2 = tp_2/(tp_2+fn_2+fp_2)
iou_3 = tp_3/(tp_3+fn_3+fp_3)
iou_4 = tp_4/(tp_4+fn_4+fp_4)
iou_0 = tp_0/(tp_0+fn_0+fp_0)
f1_1 = 2*tp_1/(2*tp_1+fn_1+fp_1)
f1_2 = 2*tp_2/(2*tp_2+fn_2+fp_2)
f1_3 = 2*tp_3/(2*tp_3+fn_3+fp_3)
f1_4 = 2*tp_4/(2*tp_4+fn_4+fp_4)
f1_0 = 2*tp_0/(2*tp_0+fn_0+fp_0)
accuracy = (tp_1+tp_2+tp_3+tp_4+tp_0)/count
f1 = (f1_1+f1_2+f1_3+f1_4+f1_0)/5
average = (iou_1+iou_2+iou_3+iou_4+iou_0)/5
p0 = accuracy
p1x0 = (ka[0][0]+ka[0][1]+ka[0][2]+ka[0][3]+ka[0][4])*(ka[0][0]+ka[1][0]+ka[2][0]+ka[3][0]+ka[4][0])
p1x1 = (ka[1][0]+ka[1][1]+ka[1][2]+ka[1][3]+ka[1][4])*(ka[0][1]+ka[1][1]+ka[2][1]+ka[3][1]+ka[4][1])
p1x2 = (ka[2][0]+ka[2][1]+ka[2][2]+ka[2][3]+ka[2][4])*(ka[0][2]+ka[1][2]+ka[2][2]+ka[3][2]+ka[4][2])
p1x3 = (ka[3][0]+ka[3][1]+ka[3][2]+ka[3][3]+ka[3][4])*(ka[0][3]+ka[1][3]+ka[2][3]+ka[3][3]+ka[4][3])
p1x4 = (ka[4][0]+ka[4][1]+ka[4][2]+ka[4][3]+ka[4][4])*(ka[0][4]+ka[1][4]+ka[2][4]+ka[3][4]+ka[4][4])
p1 = (p1x0+p1x1+p1x2+p1x3+p1x4)/(count*count)
kappa = (p0 - p1)/(1 - p1)
precision = ((tp_0/(tp_0+fp_0))+(tp_1/(tp_1+fp_1))+(tp_2/(tp_2+fp_2))+(tp_3/(tp_3+fp_3))+(tp_4/(tp_4+fp_4)))/5
recall = 1/(2/f1-1/precision)
print ("vegetation_iou+f1:", iou_1, "  ", f1_1)
print ("building_iou+f1:", iou_2, "  ", f1_2)
print ("water_iou+f1:", iou_3, "  ", f1_3)
print ("road_iou+f1:", iou_4, "  ", f1_4)
print ("others_iou+f1:", iou_0, "  ", f1_0)
print ("miou:", average)
print ("f1:",f1)
print ("accuracy:",accuracy)
print ("kappa:",kappa)
print ("precision:",precision)
print ("recall:",recall)




