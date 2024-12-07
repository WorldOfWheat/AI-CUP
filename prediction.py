print('==================== 評估模型 ====================')

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


def gettime():
    # t = 1732548594.488884, str_t = 2024-11-25 15:29:54
    t = time.time() + (8 * 3600)
    str_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
    return t, str_t


def dt(content):
    # 20240101090001
    content = str(content)
    mm = int(content[4:6])
    dd = int(content[6:8])
    HH = int(content[8:10])
    MM = int(content[10:12])
    LocationCode = int(content[12:14])
    return LocationCode, mm, dd, HH, MM


def ofea(content):
    # 20240101090001
    LocationCode, mm, dd, HH, MM = dt(content)
    md = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    f1 = sum(md[0:(mm - 1)]) + dd
    f2 = HH * 60 + MM
    sinHH, cosHH, sinf2, cosf2, sincosHH, sincosf2 = sincosHHf2(HH, f2)
    return f1, f2, sinHH, cosHH, sinf2, cosf2, sincosHH, sincosf2


def sincosHHf2(HH, f2):
    bh = 0
    eh = 24
    if HH < bh or HH > eh:
        sinHH = 0.001
        cosHH = 0.001
    else:
        sinHH = np.sin((HH - bh) * (2 * np.pi / (eh - bh)))
        cosHH = np.cos((HH - bh) * (2 * np.pi / (eh - bh)))
    bh *= 60
    eh *= 60
    if f2 < bh or f2 > eh:
        sinf2 = 0.001
        cosf2 = 0.001
    else:
        sinf2 = np.sin((f2 - bh) * (2 * np.pi / (eh - bh)))
        cosf2 = np.cos((f2 - bh) * (2 * np.pi / (eh - bh)))
    sinHH = round(sinHH, 6)
    cosHH = round(cosHH, 6)
    sinf2 = round(sinf2, 6)
    cosf2 = round(cosf2, 6)
    sincosHH = round(sinHH * cosHH, 6)
    sincosf2 = round(sinf2 * cosf2, 6)
    return sinHH, cosHH, sinf2, cosf2, sincosHH, sincosf2


def sincosmm(mm):
    sinmm = np.sin((mm) * (2 * np.pi / (12)))
    cosmm = np.cos((mm) * (2 * np.pi / (12)))
    sinmm = round(sinmm, 6)
    cosmm = round(cosmm, 6)
    return sinmm, cosmm


def sincosf1(f1):
    sinf1 = np.sin((f1) * (2 * np.pi / (365)))
    cosf1 = np.cos((f1) * (2 * np.pi / (365)))
    sinf1 = round(sinf1, 6)
    cosf1 = round(cosf1, 6)
    return sinf1, cosf1


st, str_st = gettime()
print('開始時間：' + str_st)

'''
input_dir = 
output_dir = 
'''
input_dir = '/kaggle/input/'
output_dir = '/kaggle/working/'

# 載入模型
#model_name = '20241127-100152_f0011_e0001_b0256_L0064_L0064'
regressor = load_model('/kaggle/input/test2/tensorflow2/default/1/Z_WheatherLSTM_20241205-151522_f0011_e0050_b0256_L0512_L0512.h5')

# 載入測試資料
DataName = '/kaggle/input/testdata2/upload(no answer).csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
EXquestion = SourceData['序號'].values


md = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值

count = 0
len_ex = len(EXquestion)
while (count < len_ex):
    print('count : ', str(count).zfill(8))
    sn = str(int(EXquestion[count]))
    LocationCode, mm, dd, HH, MM = dt(sn)
    inputs = []

    for i in range(HH, 17):
        for j in range(MM, 60):
            ff1 = sum(md[0:(mm - 1)]) + dd
            ff2 = i * 60 + j
            sinHH, cosHH, sinf2, cosf2, sincosHH, sincosf2 = sincosHHf2(i, ff2)
            sinmm, cosmm = sincosmm(mm)
            sinf1, cosf1 = sincosf1(ff1)
            
            fea = [LocationCode, mm, dd, i, j, ff1, ff2]
            
            fea = [LocationCode, mm, dd, i, j, ff1, ff2,
                   sinHH, cosHH, sinf2, cosf2]

            fea = [LocationCode, mm, dd, ff1, ff2,
                   cosHH, cosf2]
            
            fea = [LocationCode, mm, dd, ff1, ff2,
                   sinHH, sinf2]

            fea = [LocationCode, mm, dd, i, j, ff1, ff2,
                   sinHH, sinf2]

            fea = [LocationCode, mm, dd, i, j, ff1, ff2,
                   sinHH, cosHH, sinf2, cosf2]
            
            inputs.append(fea)

    NewTest = np.array(inputs)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 1))
    predicted = regressor.predict(NewTest)

    mv = 10
    for i in range(0, 48):
        power = np.mean(predicted[i * 10: i * 10 + mv])
        PredictOutput.append(round(power, 2))
    count += 48


Serial = list(EXquestion)
df = pd.DataFrame(Serial, columns=['序號'])
df['答案'] = PredictOutput

# 將 DataFrame 寫入 CSV 檔案
# tz = timezone(timedelta(hours=+8))
# NowDateTime = datetime.now(tz).strftime("%Y%m%d-%H%M%S")
df.to_csv(output_dir + 'Z_output_' + model_name + '.csv', index=False)
print('Output CSV File Saved')


et, str_et = gettime()
print('開始時間：' + str_st)
print('結束時間：' + str_et)
print('執行時間：' + '%.2f' % (et - st) + 's\n')
