#!/usr/bin/python
#coding:utf-8

import numpy as np
import os
import cv2

from keras.models import load_model

path = "../Face_Recognition_MLP"
os.chdir(path)

h = 32
w = 32

model = load_model('mlp_face.h5')

fpath01 = 'test'
fnames01 = os.listdir(fpath01)

# Set the initial space to save your test_data
x_test = np.zeros([len(fnames01),1024], dtype='float32')

# load test_data
inum = 0
for i in range(1,len(fnames01)+1):
    s = str(i)
    x_imgs = cv2.imread(fpath01 + '/' + s + '_test.jpg',0)
    x_imgs = x_imgs.astype('float32')/255
    x_imgs = np.reshape(x_imgs, h * w, 1)
    x_test[inum,:] = x_imgs.copy()
    inum += 1

# predict the test result	
Y = model.predict(x_test,verbose=1)
Y1 = np.where(Y > 0.9,1,0)
print Y1[0,:]
print Y1[1,:]

