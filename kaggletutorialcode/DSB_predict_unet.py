# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:37:42 2017

@author: 572203
"""

# JFB: This code is pretty poorly commented.  Adding additional comments


from __future__ import print_function

import numpy as np
import os
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from tqdm import tqdm

# Any way to pull out paths into a separate config file?  Add this to the Git Ignore file?
working_path = "E:/stage2/results/"

# Theano?! I thought they were using TensorFlow backend?
# Could be that they are using the Theano dimension ordering (z, x, y)
# rather than TensorFlow dimension ordering (x, y, z) but still using 
# TensorFlow backend.  If this is the case, the code should be refactored
# to avoid future confusion
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#Why do we have effectively the same function here twice?
def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Does this really need to be a function?  Just a simple negation correct?
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Ahh as dice_coef and dice_coef_loss are both being fed to the kera.model funtion
# as arguments, they probably do need to be their own functions



# get_unet builds the conv net architecture.  Note the overall pattern:
# - 4x conv-conv-pool
# - 1x conv-conv
# - 4x up-conv-conv

# Activation function at all layers is ReLU except for final layer

# Note how the num_filters increases 32 > 64 > 128 > 256 > 512 > 256 > 128 > 64 > 32
# This progression is likely the source of the name "u-net"

def get_unet():
    inputs = Input((1,img_rows, img_cols))

    # Note that inputs is not an argument to a function, but rather being multiplied
    # by the output of Convolution2D...

    # Syntax is Convolution2D(num_filters, num_rows_per_kernel, num_cols_per_kernel)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    # This is the output layer
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    # Simply pass the input and output layer to the model function
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

# In LUNA_train_unet, this function was train and predict. For DSB, we only want to predict.
# Let's remove all training related code
def predict():

    # We still need to stand up the unet model, if only to use it for prediction
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('C:/Users/576473/Documents/GitHub/DSB-2017/kaggletutorialcode/unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    
    patients = os.listdir(working_path)
    patients = [i for i in patients if "DSBTestImages_" in i] #patietns is a list of filenames
    patients.reverse()
    patients = patients[106:] 
    # Loop through patients
    for patient_num, patient_file in enumerate(tqdm(patients)):
        # Load in the processed DSB Images
        imgs_test = np.load(working_path + patient_file).astype(np.float32)
        # Why is imgs_test not normalized in the same way as imgs_train?  
        # Shouldn't the processing steps be consistent?
        
        imgs_test = imgs_test[1::2]

        num_test = len(imgs_test)

        # Initialize the numpy ND array to hold predicted masks
        imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
        
        # Loop through images
        for fcount, i in enumerate(tqdm(range(0,num_test))):
            imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]

        np.save(working_path + 'masksDSBTestPredicted_' + str(patient_file.split('_')[1].split('.')[0]) + '.npy', imgs_mask_test)
    
    # We can't calculate error because we don't have ground truth for the DSB image
    
if __name__ == '__main__':
    predict()