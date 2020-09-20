import cv2
import random
import numpy as np
import os
import argparse
import tensorflow as tf
import keras
import unet_train as un
import unet_res as res
import count as cn
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['1.png']

image_size = 256

classes = [0. ,  1.,  2.,   3.  , 4.]
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args
    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"],custom_objects = {'mean_iou':un.mean_iou , "lov_softmax":res.lov_softmax, "tversky_loss":res.tversky_loss, "multi_category_focal_loss2":res.multi_category_focal_loss2(gamma=2., alpha=.25),"multi_category_focal_loss2_fixed":res.multi_category_focal_loss2(gamma=2., alpha=.25), "f1":res.f1,"lov_cate":res.lov_cate})
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('./test/' + path)
        image = image.swapaxes(0, 2)
        image = image.swapaxes(1, 2)
        _, h, w = image.shape
        print('image shape: ', image.shape)
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img = np.zeros((3, padding_h, padding_w), dtype=np.uint8)
        print('src1:', padding_img.shape)
        padding_img[:, 0:h, 0:w, ] = image[:, :, :]
        print('src2:', padding_img.shape)
        padding_img = padding_img.astype("float") / 255.0
        print('src3:', padding_img.shape)
        padding_img = img_to_array(padding_img)
        padding_img = padding_img.swapaxes(0, 2)
        padding_img = padding_img.swapaxes(0, 1)
        print('src4:', padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                print('i:', i, ' j:', j, ' stride:', stride, ' image_size:', image_size)
                crop = padding_img[:3,i*stride:i*stride+image_size,j*stride:j*stride+image_size]
                print (crop.shape)
                _,ch,cw = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue

                crop = np.expand_dims(crop, axis=0)
                # print 'crop:',crop.shape
                pred = model.predict(crop,verbose = 2)
                predd = np.argmax(pred[0], axis=1)
                predd = labelencoder.inverse_transform(predd)
                # print (np.unique(pred))
                predd = predd.reshape((256, 256)).astype(np.uint8)
                # print 'pred:',pred.shape
                cn.image_count(mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] , *predd[:, :])
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = predd[:, :]

        cv2.imwrite('./predict/pre'+str(n+1)+'.png',mask_whole[0:h,0:w])
        
    

    
if __name__ == '__main__':
    args = args_parse()
    predict(args)



