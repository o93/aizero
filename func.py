# coding: utf-8

import numpy as np

from PIL import Image

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

def mean_squared_error(y, t):
  return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
  return -np.sum(t * np.log(y + 1e-7))

def save_img(img, file_path):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.save(file_path)
