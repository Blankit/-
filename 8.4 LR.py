import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
import pandas as pd

path = 'iris.data'
col = ['sepal length','sepal width','petal length','petal width','class']
data = pd.read_csv(path, header=None,names=col)
# print(data.head())

#将字符标签转成数字标签
data['class'] = pd.Categorical(data['class']).codes
label = data['class']
features = data[['sepal length','sepal width']]

# 构建模型
logisreg = LogisticRegression()
logisreg.fit(features,data['class'])
print(logisreg.intercept_,logisreg.coef_) #模型的权值和偏置
y_hat = logisreg.predict(features)# 训练样本的预测值，是数字标签，实际相当于LR做多分类
print(u'准确度：%.2f%%'%(100*np.mean(y_hat==label.ravel())))
# 划分特征值所在的区域为一个一个的方格
x_min,x_max=features['sepal length'].min()-0.1,features['sepal length'].max()+0.1
y_min,y_max=features['sepal width'].min()-0.1,features['sepal width'].max()+0.1
xx,yy = np.meshgrid(np.linspace(x_min,x_max,200),np.linspace(y_min,y_max,200))
grid_test = np.stack((xx.flat,yy.flat),axis=1)
# 计算分隔点的类别
y_predict = logisreg.predict(grid_test)
# 设置样本点和背景的颜色
cm_pt = mpl.colors.ListedColormap(['#22B14C','#ED1C24','#A0A0FF'])# 点的颜色
cm_bg = mpl.colors.ListedColormap(['#B0E0E6','#FFC0CB','#B5E61D'])# 背景颜色
# 设置坐标轴显示范围
plt.xlim(x_min-0.1,x_max+0.1)
plt.ylim(y_min-0.1,y_max+0.1)
print(y_predict)
# 划分区域的类别
plt.pcolormesh(xx,yy,y_predict.reshape(xx.shape),cmap=cm_bg)
# 显示训练样本
plt.scatter(features['sepal length'],features['sepal width'],
            c=data['class'],cmap=cm_pt,edgecolors='k', s=40)

# 区域块的图例
patchs = [mpatches.Patch(color='#B0E0E6', label='Iris-setosa'),
          mpatches.Patch(color='#FFC0CB', label='Iris-versicolor'),
          mpatches.Patch(color='#B5E61D', label='Iris-virginica')]
plt.legend(handles=patchs, fancybox=True, framealpha=0.8)
#显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel(u'萼片的长度')
plt.ylabel(u'萼片的宽度')

plt.show()
