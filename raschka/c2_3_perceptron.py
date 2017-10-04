#!/usr/bin/env python3
'''
Raschka's PML Book Chaper2_3
'''

import sys
sys.path.append('raschka')
sys.path.append('classifier')
sys.path.append('graph')


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from perceptron import Perceptron
from decision_regions import plt_decision_regions

# Data Preparation
df = pd.read_csv('raschka/data/iris.data', header=None)
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())
y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

# Scatter Plot
plt.scatter(X[:50,0],X[:50,1],color="red", marker="o", label="setosa")
plt.scatter(X[50:100,0],X[50:100,1],color="blue",marker="x", label="versicolor")
plt.xlabel("sepal length[cm]")
plt.ylabel('petal length[cm]')
plt.legend(loc="upper left")
plt.show()

# ppn fit
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.show()

# Decision Regions(ppn fit)
plt_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm')
plt.legend(loc="upper left")
plt.show()
