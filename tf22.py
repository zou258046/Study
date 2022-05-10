# -*- coding: utf-8 -*-
__author__ = 'dongfangyao'
__date__ = '2018/12/14 上午11:01'
__product__ = 'PyCharm'
__filename__ = 'tf22'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('traffic_data.csv', encoding='utf-8')
print(df.head())

# 2. 读取特征属性X 与 目标属性Y
x = df[['人口数', '机动车数', '公路面积']]
y = df[['客运量', '货运量']]

# 3. 因为x和y的数据取值范围太大了，防止梯度爆炸，所以做一个归一化操作(使用区间缩放法)
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)

# 为了后面和w进行矩阵的乘法操作
# 神经网络中 同一层之间的神经元没有关系
sample_in = x.T
sample_out = y.T
# sample_in 3*20
# sample_out 2*20

# 超参数
max_epochs = 60000
learn_rate = 0.035
mse_final = 6.5e-4
sample_number = x.shape[0]
input_number = 3
out_number = 2
hidden_unit_number = 8

# 网络参数
# 8*3的矩阵
w1 = 0.5 * np.random.rand(hidden_unit_number, input_number) - 0.1
# 8*1的矩阵
b1 = 0.5 * np.random.rand(hidden_unit_number, 1) - 0.1
# 2*8的矩阵
w2 = 0.5 * np.random.rand(out_number, hidden_unit_number) - 0.1
# 2*1的矩阵
b2 = 0.5 * np.random.rand(out_number, 1) - 0.1


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


mse_history = []
# BP的计算
for i in range(max_epochs):
    # FP过程
    # 隐藏层的输出(8*20)
    hidden_out = sigmoid(np.dot(w1, sample_in) + b1)
    # 输出层的输出（为了简化我们的写法，输出层不进sigmoid激活）（2*20）
    network_out = np.dot(w2, hidden_out) + b2

    # 错误 误差
    # 2*20
    err = sample_out - network_out
    mse = np.average(np.square(err))
    mse_history.append(mse)
    if mse < mse_final:
        break

    # BP过程
    # delta2: 2*20
    # 隐层与输出层之间的 2*20
    delta2 = -err
    # 输入层与隐层之间的
    delta1 = np.dot(w2.transpose(), delta2) * hidden_out * (1 - hidden_out)
    # w2的导数 2*8
    delta_w2 = np.dot(delta2, hidden_out.transpose())
    # b2的导数 2*1
    delta_b2 = np.dot(delta2, np.ones((sample_number, 1)))
    # w1的导数
    delta_w1 = np.dot(delta1, sample_in.transpose())
    # b1的导数
    delta_b1 = np.dot(delta1, np.ones((sample_number, 1)))
    # w2: 2*8的矩阵， 那也就是要求delta_w2必须是2*8的一个矩阵
    w2 -= learn_rate * delta_w2
    b2 -= learn_rate * delta_b2
    w1 -= learn_rate * delta_w1
    b1 -= learn_rate * delta_b1

# print('看误差是否在降低：', mse_history)

# 误差曲线图
mse_history10 = np.log10(mse_history)
min_mse = min(mse_history10)
plt.plot(mse_history10)
plt.plot([0, len(mse_history10)], [min_mse, min_mse])
ax = plt.gca()
ax.set_yticks([-2, -1, 0, 1, 2, min_mse])
ax.set_xlabel('iteration')
ax.set_ylabel('MSE')
ax.set_title('误差数值', fontdict={'fontsize': 18, 'color': 'red'})
plt.show()

# 仿真输出和实际输出对比图
# 隐藏层输出
hidden_out = sigmoid((np.dot(w1, sample_in) + b1))
# 输出层输出
network_out = np.dot(w2, hidden_out) + b2
# 反转获取实际值
network_out = y_scaler.inverse_transform(network_out.T)
sample_out = y_scaler.inverse_transform(y)

#name=['客流量','货流量']
#test=pd.DataFrame(columns=name,data=network_out)                保存数据
#print(test)
#test.to_csv('D:/pythonProject/venv/Lib/python learn/testcsv.csv')


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
line1, = axes[0].plot(network_out[:, 0], 'k', marker='o')
line2, = axes[0].plot(sample_out[:, 0], 'r', markeredgecolor='b', marker='*', markersize=9)
axes[0].legend((line1, line2), ('预测值', '实际值'), loc='upper left')
axes[0].set_title('客流量模拟 ', fontdict={'fontsize': 18, 'color': 'red'})
line3, = axes[1].plot(network_out[:, 1], 'k', marker='o')
line4, = axes[1].plot(sample_out[:, 1], 'r', markeredgecolor='b', marker='*', markersize=9)
axes[1].legend((line3, line4), ('预测值', '实际值'), loc='upper left')
axes[1].set_title('货流量模拟 ', fontdict={'fontsize': 18, 'color': 'red'})
plt.show()







