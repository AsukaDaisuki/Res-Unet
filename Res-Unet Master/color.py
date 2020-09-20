import cv2
from PIL import Image
import numpy as np
img = Image.open('C:/Users/blue_/Desktop/Satellite-Segmentation-master/unet/predict/pre1.png')
width = img.size[0]#长度
height = img.size[1]#宽度unet
new_img=Image.new('RGB', (width, height), (0,0,0))
for i in range(0,width):#遍历所有长度的点
    for j in range(0,height):#遍历所有宽度的点
        data = (img.getpixel((i, j)))
        if data == 1:#vegetation
            new_img.putpixel((i,j),(155,255,84))
        elif data == 4: #road
            new_img.putpixel((i, j), (255,191,0))
        elif data == 2:
            new_img.putpixel((i, j), (34,180,238))
        elif data == 3: #water
            new_img.putpixel((i, j), (38,71,139))
        elif data == 255:
            new_img.putpixel((i, j), (0, 0, 0))
new_img.save('tversky_final.png')



