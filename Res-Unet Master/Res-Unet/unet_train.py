#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import keras
from keras.models import Sequential  
from keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input,BatchNormalization,Activation
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array
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
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def Jaccard(y_true,y_pred):
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

def lov(y_true , y_pred ):
    loss = L.lovasz_hinge(y_pred, y_true)
    y_pred = keras.activations.sigmoid(y_pred)
    return  loss

def lov_softmax(y_true , y_pred ):
    loss = L.lovasz_softmax(y_pred, y_true,order='BHWC')
    return loss
def unet(lr):
    inputs = Input((3, img_w, img_h))

    conv1 = Conv2D(32, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(32, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(64, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Conv2D(128, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Conv2D(256, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation="elu", padding="same", kernel_initializer=keras.initializers.he_normal())(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Dropout(0.2)(conv5)
    print (conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(up6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    print (conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(up7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    print (conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(up8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)
    print (conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(up9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same",kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)
    print (conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)
    print (conv10)
    #conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)
    conv10 = Reshape((n_label,img_w*img_h))(conv10)
    conv10 = Permute((2,1))(conv10)
    #print (conv10)
    model = Model(inputs=inputs, outputs=conv10)
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=IoU_loss, metrics=['accuracy',mean_iou])
    return model


  
def train(args): 
    EPOCHS = 30
    BS = 8
    swa_start = 23
    lr_start = 0.01
    lr_end = 0.0001
    schedule = lambda epoch: LR_schedule(epoch, SWA_START=swa_start, lr_start=lr_start, lr_end=lr_end)
    lr_schedule_obj = LearningRateScheduler(schedule=schedule)
    swa_obj = SWA('', swa_start)
    model = unet(lr=lr_start)
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')
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
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.plot(np.arange(0, N), H.history["mean_iou"], label="mean_iou")
    plt.plot(np.arange(0, N), H.history["val_mean_iou"], label="val_mean_iou")
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
