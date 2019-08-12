# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:41:59 2019
@author: Yesser H. Nasser
"""

'''
building RestNet50 
'''

import keras 
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from glob import glob

from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Dense, Activation, Flatten, AveragePooling2D, Add, ZeroPadding2D, Dropout
from keras.utils import to_categorical
from keras.models import Model

# ======================================import data=======================================================
# import data
DIR = '/DataSets/SkinCancer'
#import csv
skin_cancer_df = pd.read_csv(os.path.join(DIR, 'HAM10000_metadata.csv'))
# dict of image id and image path
image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(DIR,'*','*.jpg'))}
# dict lesion
lesion_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
        }
# creat a column for image path
skin_cancer_df['image_path'] = skin_cancer_df.image_id.map(image_path_dict.get)
# creat a column for lesion_type
skin_cancer_df['lesion_type'] = skin_cancer_df.dx.map(lesion_dict.get)

# add a column for label (target)
skin_cancer_df['label'] = pd.Categorical(skin_cancer_df['lesion_type']).codes

# check for null values
print(skin_cancer_df.isnull().sum())

# fill all the null values
skin_cancer_df.age.fillna(skin_cancer_df.age.mean(), inplace=True) 

print(skin_cancer_df.dx.unique(), len(skin_cancer_df.dx.unique()))
print(skin_cancer_df.localization.unique(), len(skin_cancer_df.localization.unique()))
print(skin_cancer_df.label.unique(), len(skin_cancer_df.label.unique()))

# loading image matrix data
skin_cancer_df['image_data'] = skin_cancer_df['image_path'].map(lambda x: np.array(Image.open(x).resize((100,75))))

# plot one image of data
plt.imshow(skin_cancer_df['image_data'][0])

# ======================================================= target ====================================================================
y_target = np.asarray(skin_cancer_df.label.tolist())
y_one_hot = to_categorical(y_target)
n_classes = 7
# =================================================== features (input images) =======================================================
X = np.asarray(skin_cancer_df.image_data.tolist()).reshape(-1,75,100,3)
# change to float 32
X = X.astype('float32')
X = X/255

# ================================================ creat train and test data set # ==================================================
X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split(X,y_one_hot,test_size=0.2)

# Define Main code ResNet 50
# ======================================Building the graph ResNet50 ======================================
# define Identity Block

def Identity_Block(x,f_size,filters,stage,block):
    
    #define names
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'res'+str(stage)+block+'branch'
    # define filters
    F1, F2, F3 = filters
    #x_shortcut
    x_shortcut = x
    # x main path step a
    x = Conv2D(F1, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2a', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = BatchNormalization(axis=3, name = bn_name_base +'2a')(x)
    x = Activation('relu')(x)
    
    # x main path step b
    x = Conv2D(F2, (f_size,f_size), strides = (1,1), padding = 'same', name = conv_name_base+'2b', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = BatchNormalization(axis=3, name = bn_name_base +'2b')(x)
    x = Activation('relu')(x)    
    
    # x main path step c
    x = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = BatchNormalization(axis=3, name = bn_name_base +'2c')(x)  
    
    # adding x_shortcut and x and apply relu activation
    x = Add()([x_shortcut,x])
    x = Activation('relu')(x)
    
    return x


def Convolutional_Block(x,f_size,filters,stage,block, s):
    
    #define names
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'res'+str(stage)+block+'branch'
    # define filters
    F1, F2, F3 = filters
    #x_shortcut
    x_shortcut = x
    # x main path step a
    x = Conv2D(F1, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2a', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = BatchNormalization(axis=3, name = bn_name_base +'2a')(x)
    x = Activation('relu')(x)
    
    # x main path step b
    x = Conv2D(F2, (f_size,f_size), strides = (s,s), padding = 'same', name = conv_name_base+'2b', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = BatchNormalization(axis=3, name = bn_name_base +'2b')(x)
    x = Activation('relu')(x)    
    
    # x main path step c
    x = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = BatchNormalization(axis=3, name = bn_name_base +'2c')(x)  
    
    #x_shortcut path
    x_shortcut = Conv2D(F3, (1,1), strides=(s,s), padding = 'valid', name = conv_name_base+'1', kernel_initializer = keras.initializers.glorot_uniform())(x_shortcut)
    x_shortcut = BatchNormalization(axis =3, name = bn_name_base +'1')(x_shortcut)
    
    # adding x_shortcut and x and apply relu activation
    x = Add()([x_shortcut,x])
    x = Activation('relu')(x)
    
    return x


#define input shape
x_input = Input(shape=(75,100,3))

#define zero Padd
x = ZeroPadding2D((3,3))(x_input)

# stage 1
x = Conv2D(64, (7,7), strides=(2,2), name='conv1', kernel_initializer = keras.initializers.glorot_uniform())(x)
x = BatchNormalization(axis=3, name='bn_conv1')(x)
x = Activation ('relu')(x)
x = MaxPooling2D((3,3), strides=(2,2), name='maxpool_conv1')(x)

# stage 2
x = Convolutional_Block(x,f_size=3,filters=[64,64,256],stage=2,block='a', s=1)
x = Identity_Block(x,f_size=3,filters=[64,64,256],stage=2,block='b')
x = Identity_Block(x,f_size=3,filters=[64,64,256],stage=2,block='c')

# additional dropout and batchnormalization
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
# =================

# stage 3
x = Convolutional_Block(x,f_size=3,filters=[128,128,512],stage=3,block='a', s=2)
x = Identity_Block(x,f_size=3,filters=[128,128,512],stage=3,block='b')
x = Identity_Block(x,f_size=3,filters=[128,128,512],stage=3,block='c')
x = Identity_Block(x,f_size=3,filters=[128,128,512],stage=3,block='d')

# additional dropout and batchnormalization
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
# =================

# stage 4
x = Convolutional_Block(x,f_size=3,filters=[256,256,1024],stage=4,block='a', s=2)
x = Identity_Block(x,f_size=3,filters=[256,256,1024],stage=4,block='b')
x = Identity_Block(x,f_size=3,filters=[256,256,1024],stage=4,block='c')
x = Identity_Block(x,f_size=3,filters=[256,256,1024],stage=4,block='d')
x = Identity_Block(x,f_size=3,filters=[256,256,1024],stage=4,block='e')
x = Identity_Block(x,f_size=3,filters=[256,256,1024],stage=4,block='f')

# additional dropout and batchnormalization
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
# =================

# stage 5
x = Convolutional_Block(x,f_size=3,filters=[512,512,2048],stage=5,block='a', s=2)
x = Identity_Block(x,f_size=3,filters=[512,512,2048],stage=5,block='b')
x = Identity_Block(x,f_size=3,filters=[512,512,2048],stage=5,block='c')

# additional dropout and batchnormalization
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
# =================

# Avergae pooling
x = AveragePooling2D((2,2), name='AveragePooling')(x)
# Flatten
x = Flatten()(x)
# output layer  

# additional dropout 
x = Dropout(0.4)(x)
# =================

x = Dense(n_classes, activation = 'softmax', name = 'output_layer')(x)

#define model 
Model_Skin_Cancer = Model(x_input,x,name='ResNet50')

Model_Skin_Cancer.summary()

batch_size = 20
training_epochs = 40


Model_Skin_Cancer.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
Model_Skin_Cancer_ResNet50 = Model_Skin_Cancer.fit(X_train, y_train_one_hot, batch_size = batch_size, epochs = training_epochs, validation_data=(X_test,y_test_one_hot))
Model_Skin_Cancer.save('Model_Skin_Cancer_ResNet50.h5py')

# evalution
test_eval = Model_Skin_Cancer.evaluate(X_test,y_test_one_hot)
print('model loss is ', test_eval[0])
print('model accruracy is ', test_eval[1])


# evaluate the training

accuracy = Model_Skin_Cancer_ResNet50.history['acc']
val_accuracy = Model_Skin_Cancer_ResNet50.history['val_acc']

loss = Model_Skin_Cancer_ResNet50.history['loss']
val_loss = Model_Skin_Cancer_ResNet50.history['val_loss']

Epochs = range(len(accuracy))

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.plot(Epochs, accuracy, 'go', label = 'training accuracy')
plt.plot(Epochs, val_accuracy, 'g', label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(1)

plt.subplot(1,2,2)
plt.plot(Epochs, loss, 'ro', label = 'training loss')
plt.plot(Epochs, val_loss, 'r', label = 'validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(1)




