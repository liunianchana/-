from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import re


def RIL(x):                          # 读取列表模块 read_imgs_list
    l = os.listdir(x)                # 读取所有图片名
    del l[l.index('2_15.jpg')]       # 删除损毁的图片 2_15.jpg
    return l                         # 输出列表

# M / 2 - M / 4, N / 2 - N / 4, M / 2 + M / 4, N / 2 + N / 4
def Division(x):                     # 图像切割模块
    M, N = x.size
    box = [M / 2 - 50, N / 2 - 50, M / 2 + 50, N / 2 + 50]
    p = x.crop(box)
    return p                         # 输出切割后的图像


def var(x=None):                     #三阶矩
    mid = np.mean(((x - x.mean()) ** 3))
    return np.sign(mid) * abs(mid) ** (1/3)

def Features(x):                     # 特征提取模块
    Ft=[]
    # 切分RGB
    R,G,B = np.split(np.array(x), 3, axis=2)

    # 一阶矩
    R_1 = np.mean(R)
    G_1 = np.mean(G)
    B_1 = np.mean(B)

    # 二阶矩
    R_2 = np.std(R)
    G_2 = np.std(G)
    B_2 = np.std(B)

    # 三阶矩
    R_3 = var(R)
    G_3 = var(G)
    B_3 = var(B)

    Ft.extend([R_1,G_1,B_1,R_2,G_2,B_2,R_3,G_3,B_3])
    return Ft


def Standard(x):                     # 对总数据进行标准处理
    columnsname = ['R通道一阶矩', 'G通道一阶矩', 'B通道一阶矩', 'R通道二阶矩', 'G通道二阶矩', 'B通道二阶矩', 'R通道三阶矩', 'G通道三阶矩', 'B通道三阶矩', ]
    a = pd.DataFrame(preprocessing.normalize(x, norm='l2'),columns=columnsname)
    return a


def Changes(x):                      # 加上横坐标
    columnsname = ['R通道一阶矩', 'G通道一阶矩', 'B通道一阶矩', 'R通道二阶矩', 'G通道二阶矩', 'B通道二阶矩', 'R通道三阶矩', 'G通道三阶矩', 'B通道三阶矩', ]
    d = DataFrame(x, columns=columnsname)
    return d


def Category(x,y):                   # 加上水质类别列
    a = []
    for i in x:
        if i[0] == '1':
            a.append(1)
        elif i[0] == '2':
            a.append(2)
        elif i[0] == '3':
            a.append(3)
        elif i[0] == '4':
            a.append(4)
        else:
            a.append(5)
    y.insert(0, '水质类别', a)
    return y


def Order(x,y):                      # 加上序号列
    a = []
    b = 1
    for i in range(len(x)):
        a.append(b)
        b+=1
    y.insert(1, '序号', a)
    return y