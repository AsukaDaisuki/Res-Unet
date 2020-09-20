import cv2
from PIL import Image
import numpy as np
def image_count(mask, *crop) :
    for a in range(0,256):#遍历所有长度的点
        for b in range(0,4):#遍历所有宽度的点
            crop[a][b] = mask[a][b]


