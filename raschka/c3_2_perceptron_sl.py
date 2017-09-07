#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append('raschka')
sys.path.append('graph')
sys.path.append('classifier')


from sklearn import datasets
import numpy as np

# Datasetをロード
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
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

# エポック数40、学習率0.1でパーセプロトンのインスタンスを生成
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)
# トレーニングデータをモデルに適合させる
ppn.fit(X_train_std, y_train)

# テストデータ(X_test_std)を使って予測を実施
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test !=y_pred).sum())


# 分類の正解率を表示
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# 決定領域
from matplotlib import pyplot as plt
from decision_regions import plt_decision_regions
#トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plt_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width[standardized]')
plt.legend(loc="upper left")
plt.show()