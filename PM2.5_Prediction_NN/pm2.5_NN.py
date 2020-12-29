import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras import models
from keras import layers

df = pd.read_csv('空氣品質監測日值.csv')
print('Shape of the dataframe:', df.shape)

#   前處理－刪去空值
df['PM2.5'] = df['PM2.5'].replace('-', np.nan)
df.dropna(subset=['PM2.5'], axis=0, inplace=True)
print(df.index)
df.reset_index(drop=True, inplace=True)
print('Shape of the dataframe:', df.shape)

#   標準化成0~1
scaler_pm25 = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler_pm25.fit_transform(np.array(df['PM2.5']).reshape(-1, 1))
df['scaled_pm2.5'].head()

# 切分訓練資料及測試資料
split = len(df.index) * 0.7
df_train = df.loc[df.index < int(split)]
df_test = df.loc[df.index >= int(split)]
print('Shape of train:', df_train.shape)
print('Shape of validation:', df_test.shape)

df_train.head()
df_test.head()
df_test.reset_index(drop=True, inplace=True)

# 輸入整理(用過去七天的觀測值來預測下一天的pm2.5值）
def makeXy(ts, nb_timesteps):
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):  # start = 7 ~ stop = 33095
        X.append(list(ts.loc[i - nb_timesteps:i - 1]))  # 0~6, 1~7, 2~8, ...
        y.append(ts.loc[i])  # 7, 8, 9, ...
    X, y = np.array(X), np.array(y)
    return X, y

X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)

X_test, y_test = makeXy(df_test['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_test.shape, y_test.shape)

#   MLP
model = models.Sequential()
model.add(layers.Dense(32, activation='tanh', input_shape=(7,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='linear'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

train_history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0.2, verbose=1)

history_dict = train_history.history
history_dict.keys()

#   測試資料預測
preds = model.predict(X_test)
pred_pm25 = scaler_pm25.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# 1. 圖表顯示 accuracy
# show_train_history(train_history, 'accuracy', 'val_accuracy')
# 2. 圖表顯示 loss
# show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("")
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

#   畫圖呈現實際與預測值
Data_Ori = df_test.loc[7:56, 'PM2.5'].astype(int)
Data_Pre = pred_pm25[:50].astype(int)

plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), Data_Ori, linestyle='-', marker='*', color='r')
plt.plot(range(50), Data_Pre, linestyle='-', marker='.', color='b')
plt.legend(['Actual', 'Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')
plt.show()

aa = 0
