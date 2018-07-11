# coding: utf-8

import numpy as np
import pickle
import mnist
import func
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = mnist.load(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(len(x_batch))
print(len(t_batch))
