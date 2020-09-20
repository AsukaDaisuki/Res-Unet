#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import keras
from keras.models import Sequential  
from keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input,BatchNormalization,Activation
from keras.layers import add,Flatten, ZeroPadding2D, AveragePooling2D , SpatialDropout2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.metrics import f1_score, precision_score, recall_score
import lovasz_losses_tf as L
import torchcontrib
from swa import SWA
from lr_schedule import LR_schedule
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image  
import matplotlib.pyplot as plt
import torch.optim
import cv2
import random
import os
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7  
np.random.seed(seed)  
  
#data_shape = 360*480  
img_w = 256  
img_h = 256  
#有一个为背景  
n_label = 4+1
  
classes = [0. ,  1.,  2.,   3.  , 4.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  

image_sets = ['1.png','2.png','3.png']
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img



filepath ='../segnet/train/'

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'images'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'images/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'masks/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))
            train_label.append(label)
            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()
                train_label = labelencoder.transform(train_label)
                train_label = train_label.reshape(batch_size,256,256)
                train_label = to_categorical(train_label, num_classes=n_label)
                train_label = train_label.reshape((batch_size, img_w * img_h, n_label))
                yield (train_data,train_label)
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'images/' + url)
            img = img_to_array(img)
            valid_data.append(img)  
            label = load_img(filepath + 'masks/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()
                valid_label = labelencoder.transform(valid_label)
                valid_label = valid_label.reshape(batch_size,256, 256)
                valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size, img_w * img_h, n_label))
                yield (valid_data,valid_label)
                valid_data = []  
                valid_label = []  
                batch = 0

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)  # 将 y_true 拉伸为一维.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	1 - dice_coef(y_true, y_pred, smooth=1)


def tversky(y_true, y_pred):
    smooth = 1
    y_pred /= tf.reduce_sum(y_pred,
                            reduction_indices=len(y_pred.get_shape()) - 1,
                            keep_dims=True)
    epsilon = tf.convert_to_tensor(1e-10, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.5
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def Jaccard(y_true,y_pred):
    y_pred /= tf.reduce_sum(y_pred,
                            reduction_indices=len(y_pred.get_shape()) - 1,
                            keep_dims=True)
    epsilon = tf.convert_to_tensor(1e-10, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_true_expand = K.expand_dims(y_true, axis=0)
    y_pred_expand = K.expand_dims(y_pred, axis=-1)

    fenzi = K.dot(y_true_expand, y_pred_expand)

    fenmu_1 = K.sum(y_true, keepdims=True)

    fenmu_2 = K.ones_like(y_true_expand) - y_true_expand
    fenmu_2 = K.dot(fenmu_2, y_pred_expand)

    return K.mean((tf.constant([[1]], dtype=tf.float32) - (fenzi / (fenmu_1 + fenmu_2))), axis=-1)


def IoU_fun(eps=1e-6):
    def IoU(y_true, y_pred):
        # if np.max(y_true) == 0.0:
        #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
        intersection = K.sum(y_true * y_pred, axis=[1])
        union = K.sum(y_true, axis=[1]) + K.sum(y_pred, axis=[1]) - intersection
        #
        ious = K.mean((intersection + eps) / (union + eps), axis=0)
        return K.mean(ious)

    return IoU


def IoU_loss(y_true, y_pred):
    return 1 - IoU_fun(eps=1e-6)(y_true=y_true, y_pred=y_pred)



def iou(y_true, y_pred, label: int = 1):
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_pred /= tf.reduce_sum(y_pred,
                            reduction_indices=len(y_pred.get_shape()) - 1,
                            keep_dims=True)
    epsilon = tf.convert_to_tensor(1e-10, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)

    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels
def lov_cate(y_true, y_pred,lov_weight = 0.5):
    loss1=keras.losses.categorical_crossentropy(y_true, y_pred)
    loss2=lov_softmax(y_true, y_pred)
    return lov_weight * loss2 + (1-lov_weight)* loss1

def f1(y_true, y_pred):
 # Count positive samples.
 y_pred /= tf.reduce_sum(y_pred,
                         reduction_indices=len(y_pred.get_shape()) - 1,
                         keep_dims=True)
 epsilon = tf.convert_to_tensor(1e-10, y_pred.dtype.base_dtype)
 y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
 c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
 c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
 c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
 # If there are no true samples, fix the F1 score at 0.
 if c3 == 0:
    return 0# How many selected items are relevant?
 precision = c1/c2
 # How many relevant items are selected?
 recall = c1/c3
 # Calculate f1_score
 f1_score = 2 * (precision * recall)/(precision + recall)
 return f1_score
def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        loss2=lov_cate(y_true, y_pred)
        y_pred /= tf.reduce_sum(y_pred,
                                reduction_indices=len(y_pred.get_shape()) - 1,
                                keep_dims=True)
        epsilon = tf.convert_to_tensor(1e-10, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t + epsilon)
        weight = tf.pow(tf.subtract(1., y_t+ epsilon), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss+loss2

    return multi_category_focal_loss2_fixed


def lov_softmax(y_true , y_pred ):
    y_pred /= tf.reduce_sum(y_pred,
                            reduction_indices=len(y_pred.get_shape()) - 1,
                            keep_dims=True)
    epsilon = tf.convert_to_tensor(1e-12, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    # manual computation of crossentropy
    return L.lovasz_softmax(y_pred, y_true,order='BHWC')


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name , kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=1, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def unet(lr):
    inputs = Input((3, img_w, img_h))

    conv1 = ZeroPadding2D((3, 3))(inputs)
    conv1_1 = Conv2d_BN(conv1, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    # (128,128,64)
    conv2 = Conv_Block(conv1_1, nb_filter=64, kernel_size=(3, 3))
    conv2 = Conv_Block(conv2, nb_filter=64, kernel_size=(3, 3))
    conv2 = Conv_Block(conv2, nb_filter=64, kernel_size=(3, 3))
    # (64,64,128)
    conv3 = Conv_Block(conv2, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    conv3 = Conv_Block(conv3, nb_filter=128, kernel_size=(3, 3))
    conv3 = Conv_Block(conv3, nb_filter=128, kernel_size=(3, 3))
    conv3 = Conv_Block(conv3, nb_filter=128, kernel_size=(3, 3))
    # (32,32,256)
    conv4 = Conv_Block(conv3, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    conv4 = Conv_Block(conv4, nb_filter=256, kernel_size=(3, 3))
    conv4 = Conv_Block(conv4, nb_filter=256, kernel_size=(3, 3))
    conv4 = Conv_Block(conv4, nb_filter=256, kernel_size=(3, 3))
    conv4 = Conv_Block(conv4, nb_filter=256, kernel_size=(3, 3))
    conv4 = Conv_Block(conv4, nb_filter=256, kernel_size=(3, 3))

    # (16,16,512)
    conv5 = Conv_Block(conv4, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    conv5 = Conv_Block(conv5, nb_filter=512, kernel_size=(3, 3))
    conv5 = Conv_Block(conv5, nb_filter=512, kernel_size=(3, 3))

    conv6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    print (conv6)

    conv7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    print (conv7)

    conv8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    print (conv8)

    conv9 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)
    print (conv9)


    conv11 = Conv2D(n_label, (1, 1), activation="softmax",padding="same",kernel_initializer='he_normal')(conv9)
    print (conv11)
    #conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)
    conv11 = Reshape((n_label,img_w*img_h))(conv11)
    conv11 = Permute((2,1))(conv11)
    #print (conv10)
    model = Model(inputs=inputs, outputs=conv11)
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=lov_softmax, metrics=['accuracy',mean_iou,f1])
    return model


  
def train(args): 
    EPOCHS = 30
    BS = 8
    swa_start = 24
    lr_start = 0.001
    lr_end = 0.0005
    CYCLE = 8000
    PI=3.1415926
    #schedule = lambda x: ((lr_start - lr_end) / 2) * (np.cos(PI * (np.mod(x - 1, CYCLE) / (CYCLE))) + 1) + lr_end
    schedule = lambda epoch: LR_schedule(epoch, SWA_START=swa_start, lr_start=lr_start, lr_end=lr_end)
    lr_schedule_obj = LearningRateScheduler(schedule=schedule)
    swa_obj = SWA('', swa_start)
    model = unet(lr=lr_start)
    modelcheck = ModelCheckpoint(args['model'],monitor='val_mean_iou',save_best_only=True,mode='max')
    callable = [modelcheck,lr_schedule_obj,swa_obj]
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)
    print (valid_numb)
    print (BS)
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.plot(np.arange(0, N), H.history["val_mean_iou"], label="val_mean_iou")
    plt.plot(np.arange(0, N), H.history["val_f1"], label="val_f1")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

  

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",
                    default=True)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    filepath = '../segnet/train/'
    train(args)  
    #predict()  
