# coding: utf-8

import numpy as np
import matplotlib
import mnist
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def step(x):
  return np.array(x > 0, dtype=np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def identity(x):
  return x

def softmax(x):
  max = np.max(x)
  exp = np.exp(x - max)
  sum = np.sum(exp)
  return exp / sum

mnist.load()
