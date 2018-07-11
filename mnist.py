# coding: utf-8

import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np

url_base = "http://yann.lecun.com/exdb/mnist/"
key_train_img = "train_img"
key_train_label = "train_label"
key_test_img = "test_img"
key_test_label = "test_label"

file_names = {
  'train_img':'train-images-idx3-ubyte.gz',
  'train_label':'train-labels-idx1-ubyte.gz',
  'test_img':'t10k-images-idx3-ubyte.gz',
  'test_label':'t10k-labels-idx1-ubyte.gz'
}

dir_path = "temp/"
pkl_path = dir_path + "mnist.pkl"

train_size = 60000
test_size = 10000
img_size = 784

def _download(file_name):
  file_path = dir_path + file_name

  if os.path.exists(file_path):
    return

  print("Downloading " + file_name + " ..")
  urllib.request.urlretrieve(url_base + file_name, file_path)
  print("Done")

def _download_files():
  for file_name in file_names.values():
    _download(file_name)

def _load_label(file_name):
  file_path = dir_path + file_name

  print("Converting " + file_name + " to NumPy Array ..")
  with gzip.open(file_path, "rb") as f:
    data = np.frombuffer(f.read(), np.uint8, offset=8)
  print("Done")

  return data

def _load_img(file_name):
  file_path = dir_path + file_name
  print("Converting " + file_name + " to NumPy Array ..")
  with gzip.open(file_path, "rb") as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
  data = data.reshape(-1, img_size)
  print("Done")

  return data

def _convert_numpy():
  dataset = {}
  dataset[key_train_img] = _load_img(file_names[key_train_img])
  dataset[key_train_label] = _load_label(file_names[key_train_label])
  dataset[key_test_img] = _load_img(file_names[key_test_img])
  dataset[key_test_label] = _load_label(file_names[key_test_label])

  return dataset

def init():
  if not os.path.exists(dir_path):
    os.mkdir(dir_path)

  _download_files()
  dataset = _convert_numpy()

  print("Creating pickle file ..")
  with open(pkl_path, "wb") as f:
    pickle.dump(dataset, f, -1)
  print("Done")

def _change_one_hot_label(x):
  t = np.zeros((x.size, 10))
  for idx, row in enumerate(t):
    row[x[idx]] = 1

  return t

def load(normalize=True, flatten=True, one_hot_label=False):
  if not os.path.exists(pkl_path):
    init()

  with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)

  if normalize:
    for key in (key_train_img, key_test_img):
      dataset[key] = dataset[key].astype(np.float32)
      dataset[key] /= 255.0

  if one_hot_label:
    for key in (key_train_label, key_test_label):
      dataset[key] = _change_one_hot_label(dataset[key])

  if not flatten:
    for key in (key_train_img, key_test_img):
      dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

  return (dataset[key_train_img], dataset[key_train_label]), (dataset[key_test_img], dataset[key_test_label])

if __name__ == "__main__":
  init()
