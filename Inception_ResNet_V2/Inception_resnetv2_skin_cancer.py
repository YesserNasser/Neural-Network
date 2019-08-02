# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:17:00 2019

@author: Yesser H. Nasser
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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

# ================================================ creat train and validation data set ==============================================
#X_train, X_valid, y_train_one_hot, y_valid_one_hot = train_test_split(X_train,y_train_one_hot,test_size=0.2)

# =============================================== building the InceptionResNet v2 ===================================================
import keras 
from keras.layers import Conv2D, Dropout, Input, Dense, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, Activation, concatenate, Add, BatchNormalization
from keras.utils import to_categorical
from keras.models import Model

# define Stem in the Inception_ResNet v2
# define Stem in the Inception_ResNet v2
def Stem(x): #input (299,299,3) (modified stride to (1,1) padding to 'same' first layer 
    x = Conv2D(32,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)

    x = Conv2D(32,(3,3), padding = 'valid', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)

    x = Conv2D(64,(3,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    
    x1a = MaxPooling2D((3,3), padding = 'valid', strides =(2,2))(x)
    
    x1b = Conv2D(96,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    
    x = concatenate([x1a,x1b], axis=3)
    
    x2a = Conv2D(64,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x2a = Conv2D(64,(7,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x2a) 
    x2a = Conv2D(64,(1,7), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x2a)
    x2a = Conv2D(96,(3,3), padding = 'valid', strides =(1,1), activation='relu',kernel_initializer = keras.initializers.glorot_uniform())(x2a) 
    x2a = BatchNormalization(axis=3)(x2a)
    
    x2b = Conv2D(64,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x2b = Conv2D(96,(3,3), padding = 'valid', strides =(1,1), activation='relu',kernel_initializer = keras.initializers.glorot_uniform())(x2b)
    x2b = BatchNormalization(axis=3)(x2b)
    
    x = concatenate([x2a,x2b], axis=3)
    
    x3a = MaxPooling2D((3,3), padding = 'valid', strides =(2,2))(x)
    
    x3b = Conv2D(192,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x3b = BatchNormalization(axis=3)(x3b)
    
    x = concatenate([x3a,x3b], axis=3)
    x = Activation('relu')(x)
    
    return x # (35,35,384)

def Inception_resnet_A(x): #input (35,35,384)
    
    x_shortcut = x
    
    x1a = Conv2D(32,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1a = Conv2D(384,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1a) 
    x1a = BatchNormalization(axis=3)(x1a)
    
    x1b = Conv2D(32,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x) 
    x1b = Conv2D(32,(3,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = Conv2D(384,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = BatchNormalization(axis=3)(x1b)     
     
    x1c = Conv2D(32,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)    
    x1c = Conv2D(48,(3,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1c)     
    x1c = Conv2D(64,(3,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1c)
    x1c = Conv2D(384,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1c)
    x1c = BatchNormalization(axis=3)(x1c)     
    
    x = Add()([x_shortcut,x1a,x1b,x1c])
    x = Activation('relu')(x)
    
    return x #(35,35,384)

def Reduction_A(x,k,l,m,n): # input (35,35,384)
    
    x1a = Conv2D(k,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1a = Conv2D(l,(3,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1a)
    x1a = Conv2D(m,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1a) #(17,17,384)
    x1a = BatchNormalization(axis=3)(x1a)
    
    
    x1b = Conv2D(n,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x) #(17,17,384)
    x1b = BatchNormalization(axis=3)(x1b)

    x1c = MaxPooling2D((3,3), padding = 'valid', strides =(2,2))(x) #(17,17,384)
    
    x = concatenate([x1a,x1b,x1c], axis=3)
   
    return x #(17,17,1152)

def Inception_resnet_B(x): # input (17,17,1152)    ''' had to change 1154 to 1152 to be consistent with dimentions - Figure 17 - paper ''' 
    
    x_shortcut = x
    
    x1a = Conv2D(192,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1a = Conv2D(1152,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1a)
    x1a = BatchNormalization(axis=3)(x1a)
    
    x1b = Conv2D(128,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1b = Conv2D(160,(1,7), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = Conv2D(192,(7,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = Conv2D(1152,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = BatchNormalization(axis=3)(x1b)  
    
    x = Add()([x_shortcut,x1a,x1b])
    x = Activation('relu')(x)
    
    return x # (17,17,1152)

def Reduction_B(x): # input (17,17,1152)
    
    x1a = MaxPooling2D((3,3), padding = 'valid', strides =(2,2))(x)
    
    x1b = Conv2D(256,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1b = Conv2D(384,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1b) 
    x1b = BatchNormalization(axis=3)(x1b)
    
    x1c = Conv2D(256,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1c = Conv2D(288,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1c) 
    x1c = BatchNormalization(axis=3)(x1c)
    
    x1d = Conv2D(256,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1d = Conv2D(288,(3,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1d) 
    x1d = Conv2D(320,(3,3), padding = 'valid', strides =(2,2), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1d) 
    x1d = BatchNormalization(axis=3)(x1d)

    x = concatenate([x1a,x1b,x1c,x1d], axis=3)
    
    return x #  (8,8,2144)

def Inception_resnet_C(x): # input (8,8,2144)   ''' had to cnage 2048 to 2144 to be consistent with dimentions - Figure 19 - paper'''  

    x_shortcut = x
    
    x1a = Conv2D(192,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1a = Conv2D(2144,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1a)
    x1a = BatchNormalization(axis=3)(x1a)
    
    x1b = Conv2D(192,(1,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x1b = Conv2D(224,(1,3), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = Conv2D(256,(3,1), padding = 'same', strides =(1,1), activation='relu', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = Conv2D(2144,(1,1), padding = 'same', strides =(1,1), activation='linear', kernel_initializer = keras.initializers.glorot_uniform())(x1b)
    x1b = BatchNormalization(axis=3)(x1b)  
    
    x = Add()([x_shortcut,x1a,x1b])
    x = Activation('relu')(x)
    
    return x #  (8,8,2144)

x_input = Input(shape=(75,100,3))

#Apply Stem
x = Stem(x_input)

# Apply Icepltion A x5
x = Inception_resnet_A(x)
x = Inception_resnet_A(x)
x = Inception_resnet_A(x)
x = Inception_resnet_A(x)    
x = Inception_resnet_A(x)
# parameters for Reduction_A
k=256
l=256
m=384
n=384
# Apply Reduction A
x = Reduction_A(x,k,l,n,m)

# Apply Inception RestNet B x10
x = Inception_resnet_B(x)
x = Inception_resnet_B(x)  
x = Inception_resnet_B(x)
x = Inception_resnet_B(x) 
x = Inception_resnet_B(x)
x = Inception_resnet_B(x) 
x = Inception_resnet_B(x)
x = Inception_resnet_B(x) 
x = Inception_resnet_B(x)
x = Inception_resnet_B(x) 

# Apply Reduction B    
x = Reduction_B(x)

# Apply Inception ResNet C x5
x = Inception_resnet_C(x)    
x = Inception_resnet_C(x)
x = Inception_resnet_C(x)
x = Inception_resnet_C(x)
x = Inception_resnet_C(x)    
 
# AveragePooling   
x = GlobalAveragePooling2D(name='Average_pooling')(x)

# Dropout
x = Dropout(0.2)(x)

# output layer    
x = Dense(n_classes, activation = 'softmax')(x)

Model_Skin_Cancer = Model(x_input, x, name='Inception_ResNetv2')

Model_Skin_Cancer.summary()

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)

Model_Skin_Cancer.compile(loss = keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

# lerning rate 
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', factor = 0.5, patience = 3, verbose=1, min_lr = 0.00001)

# data augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False,
        rotation_range=10, # random rotation 
        width_shift_range=0.1, # width shift range
        height_shift_range=0.1, # height shift range 
        horizontal_flip=True, # horizontal flip
        vertical_flip=True ) # vertical flip
datagen.fit(X_train)

# fitting /training the model
training_epochs= 40
batch_size = 10

# specify data for training and validation
Model_Skin_Cancer_InceptionResNetv2 = Model_Skin_Cancer.fit_generator(datagen.flow(X_train,y_train_one_hot, batch_size = batch_size), 
                                                                                    epochs = training_epochs, 
                                                                                    validation_data = (X_test,y_test_one_hot),
                                                                                    steps_per_epoch = X_train.shape[0]/batch_size,
                                                                                    callbacks=[learning_rate_reduction])

# save the model
Model_Skin_Cancer.save('Model_Skin_Cancer_InceptionResNetv2.h5py')

## evalution
#test_eval = Model_Skin_Cancer.evaluate(X_test,y_test_one_hot)
#print('model loss is ', test_eval[0])
#print('model accruracy is ', test_eval[1])


# evaluate the training

accuracy = Model_Skin_Cancer_InceptionResNetv2.history['acc']
val_accuracy = Model_Skin_Cancer_InceptionResNetv2.history['val_acc']

loss = Model_Skin_Cancer_InceptionResNetv2.history['loss']
val_loss = Model_Skin_Cancer_InceptionResNetv2.history['val_loss']

Epochs = range(len(accuracy))

plt.figure()
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



