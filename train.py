print('==================== 訓練模型 ====================')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timezone, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization


def gettime():
    # t = 1732548594.488884, str_t = 2024-11-25 15:29:54
    t = time.time() + (8 * 3600)
    str_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
    return t, str_t


def normalize(train):
    maxf = np.max(train, axis=0)
    minf = np.min(train, axis=0)
    meanf = np.mean(train, axis=0)
    train_norm = (train - meanf) / (maxf - minf)
    return train_norm, maxf, minf, meanf


st, str_st = gettime()
print('開始時間：' + str_st)

# input_dir = ''
# output_dir = ''

input_dir = '/kaggle/input/'
output_dir = '/kaggle/working/'


df = pd.read_csv(output_dir + 'data01.csv', header=0)

for i in range(0, 6):
    df = df[df['HH'] != i]
for i in range(20, 24):
    df = df[df['HH'] != i]

x = df[['LocationCode', 'mm', 'dd', 'HH', 'MM', 'f1', 'f2']].values

x = df[['LocationCode', 'mm', 'dd', 'HH', 'MM', 'f1', 'f2',
        'sinHH', 'cosHH', 'sinf2', 'cosf2']].values

x = df[['LocationCode', 'mm', 'dd', 'f1', 'f2',
        'cosHH', 'cosf2']].values

x = df[['LocationCode', 'mm', 'dd', 'f1', 'f2',
        'sinHH', 'sinf2']].values

x = df[['LocationCode', 'mm', 'dd', 'HH', 'MM', 'f1', 'f2',
        'sinHH', 'sinf2']].values

x = df[['LocationCode', 'mm', 'dd', 'HH', 'MM', 'f1', 'f2',
        'sinHH', 'cosHH', 'sinf2', 'cosf2']].values

#train_norm, maxf, minf, meanf = normalize(x)
#x = train_norm

y = df[['Power(mW)']].values

X_train = []
y_train = []

for i in range(0, len(y)):
    X_train.append(x[i, :])
    y_train.append(y[i, :])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(
    X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train:", np.shape(X_train))
print("y_train:", np.shape(y_train))

# %%
# ==================== 建置&訓練模型 ====================
# 建置LSTM模型
epochs = 50
batch_size = 256
L1 = 512
L2 = 512

regressor = Sequential()
# 輸入層
regressor.add(LSTM(units=L1, return_sequences=True,
              input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.2))
'''
regressor.add(LSTM(units=L2, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=L2, return_sequences=True))
regressor.add(Dropout(0.2))
'''
regressor.add(LSTM(units=L2, return_sequences=False))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.2))

# 輸出層
regressor.add(Dense(units=1, activation='relu'))
regressor.compile(optimizer='adam', loss='mean_absolute_error')

# 開始訓練
history = regressor.fit(
    X_train, y_train, validation_split=0.0, epochs=epochs, batch_size=batch_size)

# 保存模型
fea_size = X_train.shape[1]
tz = timezone(timedelta(hours=+8))
NowDateTime = datetime.now(tz).strftime("%Y%m%d-%H%M%S")

model_name = NowDateTime + \
            '_f' + str(fea_size).zfill(4) + \
            '_e' + str(epochs).zfill(4) + \
            '_b' + str(batch_size).zfill(4) + \
            '_L' + str(L1).zfill(4) + \
            '_L' + str(L2).zfill(4)

regressor.save(output_dir + 'Z_WheatherLSTM_' + model_name + '.h5')
print('Model Saved')
print("model_name:" + model_name)


loss = history.history['loss']
epochs = list(range(1, len(loss) + 1))
plt.figure(dpi=300)
plt.plot(epochs, loss, 'b-o', label='train loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 200)
plt.legend(loc='upper right')
plt.grid()
plt.savefig(output_dir + 'Z_Loss_' + model_name + '.png', dpi=300, transparent=False)
plt.show()
print('Loss Saved')


et, str_et = gettime()
print('開始時間：' + str_st)
print('結束時間：' + str_et)
print('執行時間：' + '%.2f' % (et - st) + 's\n')
