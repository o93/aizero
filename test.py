# coding: utf-8

import numpy as np
import pickle
import matplotlib
import mnist
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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

def save_img(img, file_path):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.save(file_path)

def load_data():
  (x_train, t_train), (x_test, t_test) = mnist.load(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test

def load_network():
  with open("temp/sample_weight.pkl", "rb") as f:
    network = pickle.load(f)

  return network

def predict(network, x):
  W1, W2, W3 = network["W1"], network["W2"], network["W3"]
  b1, b2, b3 = network["b1"], network["b2"], network["b3"]

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  return y

x, t = load_data()
network = load_network()

accuracy_cnt = 0
for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y)

  if p == t[i]:
    accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
