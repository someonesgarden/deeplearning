#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
sys.path.append('raschka')
sys.path.append('graph')
sys.path.append('classifier')

from sklearn import datasets
import numpy as np

#Datasetをロード
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# トレーニングデータとテストデータに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 　トレーニングデータの平均と標準偏差を計算し特徴量を標準化する
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Support Vector Machine
# from sklearn.svm import SVC
# svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm.fit(X_train_std, y_train)

#Another pettern
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss="perceptron") # Perceptron
lr = SGDClassifier(loss="log") # Logistic Regression
svm = SGDClassifier(loss='hinge') # Support Vector Machine
svm.fit(X_train_std, y_train)

# 決定領域
from matplotlib import pyplot as plt
from decision_regions import plt_decision_regions
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train, y_test))

plt_decision_regions(X=X_combined_std,y=y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel("petal length[standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()