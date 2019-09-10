# coding: utf-8

import numpy as np
import func

class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None

  def forward(self, x, t):
    self.t = t
    self.y = func.softmax(x)
    self.loss = func.cross_entropy_error(self.y, self.t)

    return self.loss

  def backward(self, dout):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx
