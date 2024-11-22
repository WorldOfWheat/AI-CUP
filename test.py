print('==================== 資料前處理 ====================')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gettime():
    # t = 1732548594.488884, str_t = 2024-11-25 15:29:54
    t = time.time() + (8 * 3600)
    str_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
    return t, str_t


def dt(content):
    # 2024-01-01 06:31:08.000
    mm = int(content[5:7])
    dd = int(content[8:10])
    HH = int(content[11:13])
    MM = int(content[14:16])
    return mm, dd, HH, MM


def ofea(content):
    # 2024-01-01 06:31:08.000
    mm, dd, HH, MM = dt(content)
    md = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    f1 = sum(md[0:(mm - 1)]) + dd
    f2 = HH * 60 + MM
    sinHH, cosHH, sinf2, cosf2, sincosHH, sincosf2 = sincosHHf2(HH, f2)
    sinmm, cosmm = sincosmm(mm)
    sinf1, cosf1 = sincosf1(f1)
    return f1, f2, sinHH, cosHH, sinf2, cosf2, sinmm, cosmm, sinf1, cosf1, sincosHH, sincosf2


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

input_dir = ''
output_dir = ''
'''
input_dir = '/kaggle/input/solarenergy-20241124/'
output_dir = '/kaggle/working/'
'''

df = pd.DataFrame()
for i in range(1, 18):
    filename = input_dir + '36_TrainingData/L' + \
        str(i) + '_Train.csv'
    data = pd.read_csv(filename, usecols=[
                       'LocationCode', 'DateTime', 'Power(mW)'], header=0)
    data.shape
    data.head(5)
    data[['mm', 'dd', 'HH', 'MM']] = data.apply(
        lambda x: dt(x['DateTime']), axis=1, result_type='expand')
    data[['f1', 'f2', 'sinHH', 'cosHH', 'sinf2', 'cosf2',
          'sinmm', 'cosmm', 'sinf1', 'cosf1',
          'sincosHH', 'sincosf2']] = data.apply(lambda x: ofea(
              x['DateTime']), axis=1, result_type='expand')
    del data['DateTime']
    df = pd.concat([df, data])

d2 = [2, 4, 7, 8, 9, 10, 12]
for i in d2:
    filename = input_dir + '36_TrainingData_Additional_V2/L' + \
        str(i) + '_Train_2.csv'
    data = pd.read_csv(filename, usecols=[
                       'LocationCode', 'DateTime', 'Power(mW)'], header=0)
    data.shape
    data.head(5)
    data[['mm', 'dd', 'HH', 'MM']] = data.apply(
        lambda x: dt(x['DateTime']), axis=1, result_type='expand')
    data[['f1', 'f2', 'sinHH', 'cosHH', 'sinf2', 'cosf2',
          'sinmm', 'cosmm', 'sinf1', 'cosf1',
          'sincosHH', 'sincosf2']] = data.apply(lambda x: ofea(
              x['DateTime']), axis=1, result_type='expand')
    del data['DateTime']
    df = pd.concat([df, data])

d = df.pop('Power(mW)')
df.insert(df.shape[1], 'Power(mW)', d)
df.to_csv(output_dir + 'data01.csv', index=0)
print('Data Saved')


et, str_et = gettime()
print('開始時間：' + str_st)
print('結束時間：' + str_et)
print('執行時間：' + '%.2f' % (et - st) + 's\n')