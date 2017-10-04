#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append('raschka')
sys.path.append('graph')
sys.path.append('classifier')


from sklearn import datasets
import numpy as np

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:, 1]>0)
y_xor = np.where(y_xor, 1, -1)

from matplotlib import pyplot as plt

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='x', label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc="best")
plt.show()

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor,y_xor)


from matplotlib import pyplot as plt
from decision_regions import plt_decision_regions

plt_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc="upper left")
plt.show()
