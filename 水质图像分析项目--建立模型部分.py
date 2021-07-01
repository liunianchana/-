import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plt_show_matrics import PSM




#_standard
data = pd.read_csv('D:/result/data.csv',encoding = 'gbk')
x = data[data.columns[2:11]]
y = data['水质类别']


# 划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x*30,y,test_size=0.2,random_state=42)


#模型预处理，进行数据标准化
from sklearn.preprocessing import StandardScaler
model = StandardScaler().fit(x_train)
x_train = model.transform(x_train)
x_test = model.transform(x_test)



# 构建分类模型
from sklearn.svm import SVC
model = SVC().fit(x_train, y_train)



import pickle
pickle.dump(model,open('D:/result/svm.model','wb'))#保存模型


# 调用模型
model = pickle.load(open('D:/result/svm.model','rb'))
x_train_s_m = model.predict(x_train)
x_test_s_m = model.predict(x_test)


#准确率
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,x_train_s_m))
print(accuracy_score(y_test,x_test_s_m))



# 生成混淆矩阵并保存
from sklearn import metrics
metrics_train = metrics.confusion_matrix(y_train,x_train_s_m)         #训练样本的混淆矩阵
metrics_test = metrics.confusion_matrix(y_test,x_test_s_m)            #测试样本的混淆矩阵
print(metrics_train)
print(metrics_test)


# 保存结果
pd.DataFrame(metrics_train,index = range(0,5),columns = range(0,5)).to_csv('D:/result/metrics_train.csv')
pd.DataFrame(metrics_test,index = range(0,5),columns = range(0,5)).to_csv('D:/result/metrics_test.csv')






PSM(metrics_train,'metrics_train')
#PSM(metrics_test,'metrics_test')


