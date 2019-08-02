# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 07:37:36 2018

@author: Yesser H. Nasser
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import keras

from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, concatenate, Input, GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.layers.core import Layer


from glob import glob


# import data
DIR = 'C:/Users/Yesser/Desktop/python/DataSets/SkinCancer'
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

# build inception Module
def Inception_Module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_1x1_pool,
                     name=None):
    conv_1x1 = Conv2D(filters_1x1, (1,1), padding= 'same', activation='relu', kernel_initializer=kernel_init, bias_initializer = bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1,1), padding = 'same', activation='relu', kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3,3), padding = 'same', activation='relu', kernel_initializer = kernel_init, bias_initializer = bias_init)(conv_3x3)
    
    conv_5x5 = Conv2D(filters_5x5_reduce, (1,1), padding = 'same', activation = 'relu', kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5,5), padding = 'same', activation = 'relu', kernel_initializer = kernel_init, bias_initializer = bias_init)(conv_5x5)
    
    max_pool = MaxPool2D((3,3), padding = 'same', strides = (1,1))(x)
    max_pool = Conv2D(filters_1x1_pool, (1,1), padding ='same', activation = 'relu', kernel_initializer=kernel_init, bias_initializer = bias_init) (max_pool)
    
    output_layer = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis = 3, name = name)
    
    return output_layer

# intializers:
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.constant(value=0.2)


# define input
input_layer = Input(shape=(75,100,3))

# ==== conv 1 
x = Conv2D(64, (7,7), padding ='same', strides = (1,1), activation='relu', name='con2D_1_7x7/2', kernel_initializer=kernel_init, bias_initializer = bias_init)(input_layer)
# === maxpool 1
x = MaxPool2D((3,3), strides=(2,2), padding = 'same', name='max_pool_1_3x3/2')(x)
# === conv 2a,b
x = Conv2D(64, (1,1), padding = 'same', strides=(1,1), activation='relu', name='conv2D_2a_1x1/1')(x)
x = Conv2D(192, (3,3), padding = 'same', strides=(1,1), activation='relu', name='conv2D_2b_3x3/1')(x)     
# === Maxpool 2
x = MaxPool2D((3,3), padding = 'same', strides=(2,2), name='max_pool_2_3x3/2')(x)
# === inception 3a
x = Inception_Module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_1x1_pool=32,
                     name = 'Inception_3a')
# inception 3b
x = Inception_Module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_1x1_pool=64,
                     name = 'Inception_3b')
# maxpool 3
x = MaxPool2D((3,3), padding='same', strides=(2,2), name='max_pool_3_3x3/2')(x)
# === inception 4a
x = Inception_Module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_1x1_pool=64,
                     name = 'Inception_4a')  
# auxiliary output 1
x1 = AveragePooling2D((5,5),strides = 3)(x)
x1 = Conv2D(128,(1,1), activation='relu', padding='same')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(n_classes, activation='softmax', name='Auxiliary_output1')(x1)

# === inception 4b
x = Inception_Module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_1x1_pool=64,
                     name = 'Inception_4b')     
    
# === inception 4c
x = Inception_Module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_1x1_pool=64,
                     name = 'Inception_4c')       
# === inception 4d
x = Inception_Module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_1x1_pool=64,
                     name = 'Inception_4d')       
# === Auxiliary output 2
x2 = AveragePooling2D((5,5), strides=3)(x)
x2 = Conv2D(128, (1,1), strides=(1,1), padding='same')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(n_classes, activation='softmax', name='Auxiliary_ouput2')(x2)

# === inception 4e
x = Inception_Module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_1x1_pool=128,
                     name = 'Inception_4e')     
# === nmaxpool4
x = MaxPool2D((3,3), strides=(2,2), name='max_pool_4_3x3/2')(x)

# inception 5a
x = Inception_Module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_1x1_pool=128,
                     name = 'Inception_5a') 
# inception 5b
x = Inception_Module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_1x1_pool=128,
                     name = 'Inception_5b') 
# === GlobalAveragePooling
x = GlobalAveragePooling2D(name='AveragePool_7x7/1')(x)

# === dropout
x = Dropout(0.4)(x)

# === output layer
x = Dense(n_classes, activation='softmax', name='Final_output_layer')(x)

Model_Skin_Cancer = Model(input_layer, [x,x1,x2], name='inception_v1')

Model_Skin_Cancer.summary()

training_epochs = 40
batch_size = 20

Model_Skin_Cancer.compile(loss=['categorical_crossentropy', 'categorical_crossentropy','categorical_crossentropy'], loss_weights=[1,0.3,0.3], optimizer = keras.optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True), metrics=['accuracy'])
Model_Skin_Cancer_Inceptionv1 = Model_Skin_Cancer.fit(X_train, [y_train_one_hot, y_train_one_hot, y_train_one_hot], epochs=training_epochs, batch_size = batch_size, validation_data=(X_test,[y_test_one_hot,y_test_one_hot,y_test_one_hot]))

Model_Skin_Cancer.save('Model_Skin_Cancer_Inceptionv1.h5py')

# evalution
test_eval = Model_Skin_Cancer.evaluate(X_test,[y_test_one_hot, y_test_one_hot, y_test_one_hot])
print('model loss is ', test_eval[0])
print('model accruracy is ', test_eval[1])


# evaluate the training
accuracy_Final_output_layer = Model_Skin_Cancer_Inceptionv1.history['Final_output_layer_acc']
accuracy_Auxiliary_output1_layer = Model_Skin_Cancer_Inceptionv1.history['Auxiliary_output1_acc']
accuracy_Auxiliary_output2_layer = Model_Skin_Cancer_Inceptionv1.history['Auxiliary_output2_acc']

val_accuracy_Final_output_layer = Model_Skin_Cancer_Inceptionv1.history['val_Final_output_layer_acc']
val_accuracy_Auxiliary_output1_layer = Model_Skin_Cancer_Inceptionv1.history['val_Auxiliary_output1_acc']
val_accuracy_Auxiliary_output2_layer = Model_Skin_Cancer_Inceptionv1.history['val_Auxiliary_output2_acc']

loss_Final_output_layer = Model_Skin_Cancer_Inceptionv1.history['Final_output_layer_loss']
loss_Auxiliary_output1_layer = Model_Skin_Cancer_Inceptionv1.history['Auxiliary_output1_loss']
loss_Auxiliary_output2_layer = Model_Skin_Cancer_Inceptionv1.history['Auxiliary_output2_loss']

val_loss_Final_output_layer = Model_Skin_Cancer_Inceptionv1.history['val_Final_output_layer_loss']
val_loss_Auxiliary_output1_layer = Model_Skin_Cancer_Inceptionv1.history['val_Auxiliary_output1_loss']
val_loss_Auxiliary_output2_layer = Model_Skin_Cancer_Inceptionv1.history['val_Auxiliary_output2_loss']

Epochs = range(len(accuracy_Final_output_layer))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(Epochs, accuracy_Final_output_layer, 'g', label = 'training accuracy final layer')
plt.plot(Epochs, val_accuracy_Final_output_layer, 'go', label = 'validation accuracy final layer')
plt.plot(Epochs, accuracy_Auxiliary_output1_layer, 'm', label = 'training accuracy auxiliary output1 layer')
plt.plot(Epochs, val_accuracy_Auxiliary_output1_layer, 'mo', label = 'validation accuracy auxiliary output1 layer')
#plt.plot(Epochs, accuracy_Auxiliary_output2_layer, 'ro', label = 'training accuracy auxiliary output2 layer')
#plt.plot(Epochs, val_accuracy_Auxiliary_output2_layer, 'r', label = 'validation accuracy auxiliary output2 layer')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(1)

plt.subplot(1,2,2)
plt.plot(Epochs, loss_Final_output_layer, 'r', label = 'training loss final layer')
plt.plot(Epochs, val_loss_Final_output_layer, 'ro', label = 'validation loss final layer')
plt.plot(Epochs, loss_Auxiliary_output1_layer, 'y', label = 'training loss auxiliary output1 layer')
plt.plot(Epochs, val_loss_Auxiliary_output1_layer, 'yo', label = 'validation loss auxiliary output1 layer')
#plt.plot(Epochs, loss_Auxiliary_output2_layer, 'mo', label = 'training loss auxiliary output2 layer')
#plt.plot(Epochs, val_loss_Auxiliary_output2_layer, 'm', label = 'validation loss auxiliary output2 layer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(1)
