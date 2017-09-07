#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

M = 2
K = 3
n = 100
N = n * K

X1 = np.random.randn(n, M) + np.array([0,10])
X2 = np.random.randn(n, M) + np.array([5,5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1,0,0] for i in range(n)])
Y2 = np.array([[0,1,0] for i in range(n)])
Y3 = np.array([[0,0,1] for i in range(n)])

X = np.concatenate((X1,X2,X3),axis=0)
Y = np.concatenate((Y1,Y2,Y3),axis=0)
