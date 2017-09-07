#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
sys.path.append('raschka')
sys.path.append('classifier')
sys.path.append('graph')

import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from adaline import AdalineGD
from decision_regions import plt_decision_regions

# Data Preparation
df = pd.read_csv('raschka/data/iris.data',header=None)
y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

# 標準化(Zvalue)
X_std = np.copy(X)

X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X_std, y)
ax[0].plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title("Adaline - Learning rate 0.01")
ax[0].set_xlim([0,16])

ada2 = AdalineGD(n_iter=10, eta=0.001).fit(X_std, y)
ax[1].plot(range(1,len(ada2.cost_)+1), np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.001')
ax[1].set_xlim([0,16])

ada3 = AdalineGD(n_iter=10, eta=0.0001).fit(X_std, y)
ax[2].plot(range(1,len(ada3.cost_)+1), np.log10(ada3.cost_),marker='o')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Sum-squared-error')
ax[2].set_title('Adaline - Learning rate 0.0001')

plt.show()

# Desicion Regions
plt_decision_regions(X_std, y, classifier=ada2)
plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')
plt.legend(loc='upper left')

plt.show()