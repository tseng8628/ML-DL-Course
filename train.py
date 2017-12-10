#!/usr/bin/python
#coding:utf-8

import numpy as np
import keras
import os
import cv2

from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import optimizers


path = "../Face_Recognition_MLP"
os.chdir(path)

fpath01 = 'train'
fnames01 = os.listdir(fpath01)

# Set the initial value, can be changed by yourself 
batch_size = 5
epochs = 100
learning_rate = 0.001
h = 32					
w = 32					

# Set the initial space to save your train_data 
x_train = np.zeros([len(fnames01),1024], dtype='float32')

# Load train_data 
inum = 0
for i in range(1,len(fnames01)+1):
    s = str(i)
    x_imgs = cv2.imread(fpath01 + '/' + s + '.jpg',0)
    x_imgs = x_imgs.astype('float32')/255
    x_imgs = np.reshape(x_imgs, h * w, 1)
    x_train[inum,] = x_imgs.copy()
    inum += 1

# Set target_data to one hot type
y_train = np.array(range(0,19))
y_train = keras.utils.to_categorical(y_train, 19)

# Construct MLP neural network with Keras	
model = Sequential()
model.add(Dense(512,activation='relu',input_shape = (1024,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(19,activation='softmax'))

# We add metrics to get more results you want to see
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=learning_rate),metrics=['accuracy'])

# Another way to train the model
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)

# save model
model.save('mlp_face.h5')

# save model structure as a picture and show summary
plot_model(model, show_shapes=True,to_file='../model.png')
model.summary()