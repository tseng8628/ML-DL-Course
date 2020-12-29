import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

inputData = pd.read_csv('Input-Data.csv')
inputData_Value = inputData[['RSI', 'W%R', 'SMA', 'CCI', 'CMO', 'ROC']][0:1000]
inputData_Value2 = inputData[['RSI', 'W%R', 'SMA', 'CCI', 'CMO', 'ROC']][1000:]

targetData = pd.read_csv('target150.csv')
targetValue = targetData['Strategy']

h = 15
w = 6
offset = 1
num = len(inputData_Value)-15+1

X_train = np.zeros([num, h, w], dtype='float32')        # 訓練資料
Y_train = np.zeros([num, ], dtype='float32')           # 訓練標籤

# 從index121
for i in range(len(inputData_Value)-14):
    data = np.array(inputData_Value[i:i+15])
    X_train[i, :, :] = data.copy()
    Y_train[i] = targetValue[i+14]

X_train = X_train.reshape(X_train.shape[0], h, w, 1)
Y_train = np_utils.to_categorical(Y_train, 3)

num2 = len(inputData_Value2)-15+1
X_test = np.zeros([num2, h, w], dtype='float32')        # 訓練資料
Y_test = np.zeros([num2, ], dtype='float32')           # 訓練標籤

for i in range(len(inputData_Value2)-14):
    data = np.array(inputData_Value2[i:i+15])
    X_test[i, :, :] = data.copy()
    Y_test[i] = targetValue[i+14]

X_test = X_test.reshape(X_test.shape[0], h, w, 1)
Y_test_org = Y_test.copy()
Y_test = np_utils.to_categorical(Y_test, 3)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(h, w, 1), activation='relu', use_bias=True))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', use_bias=True))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(X_train, Y_train, validation_split=0.2, epochs=20, batch_size=300, verbose=1)

model.save('CNN-train.h5')
model.save_weights('CNN-train.weight')

history_dict = train_history.history
history_dict.keys()

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# 1. 圖表顯示 accuracy
show_train_history(train_history, 'accuracy', 'val_accuracy')
# 2. 圖表顯示 loss
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(X_test, Y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("")
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

prediction = model.predict_classes(X_test)
Confusion_matrix = pd.crosstab(Y_test_org, prediction, rownames=['labels'], colnames=['predict'])
#
# with open('Confusion_matrix.csv', 'w', newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(['RSI', 'W%R', 'SMA', 'CCI', 'CMO', 'ROC'])
#
#     for items in Confusion_matrix:
#         writer.writerow(items)

aa = 0

