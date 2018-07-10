#coding=utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *
from tensorflow.contrib import rnn

class BiLSTM(CommonModelFunc):

  def __init__(self, FLAGS, insCNNModel, insDataPro):
    self.FLAGS = FLAGS
    self.insCNNModel = insCNNModel

  # Get a graph for bidirectional LSTM
  def getBiLSTM(self):
    # Network Parameters
    num4Input = self.FLAGS.embeddingDimension  # Same as the shape of output from hROIPooling
    timeSteps = self.FLAGS.num4Features4Instance  # Same as number of features for each instance
    num4Hidden4LSTM = self.FLAGS.num4Hidden4LSTM  # The dimensionality of hidden output
    num4Classes = self.FLAGS.num4Classes  # The number of ATC classes

    # ===== LSTM layer =====
    with tf.variable_scope("lstmLayer"):
      name4W, name4B = "wOutput4LSTM", "bOutput4LSTM"
      name4Z, name4H = "zOutput4LSTM", "hOutput4LSTM"

      wOutput4LSTM = self.init_weight_variable(
          name4W,
          [num4Hidden4LSTM,
           num4Classes])

      bOutput4LSTM = self.init_bias_variable(
          name4B,
          [num4Classes])

      ###################################
      # Define lstm cells with tensorflow
      # Forward direction cell
      # Backward direction cell
      ###################################
      lstmFwCell = rnn.BasicLSTMCell(num4Hidden4LSTM, forget_bias = 1.0)
      lstmFwCell = rnn.BasicLSTMCell(num4Hidden4LSTM, forget_bias = 1.0)

      # Get lstm cell output
      try:
          self.hiddenOutputs, _, _ = rnn.static_bidirectional_rnn(
              lstmFwCell,
              lstmBwCell,
              self.insCNNModel.hROIPooling,
              dtype=tf.float32)
      except Exception:  # Old TensorFlow version only returns outputs not states
          self.hiddenOutputs = rnn.static_bidirectional_rnn(
              lstmFwCell,
              lstmBwCell,
              self.insCNNModel.hROIPooling,
              dtype=tf.float32)

      self.zOutput4LSTM = tf.add(
          tf.matmul(
              self.hiddenOutputs[-1],
              wOutput4LSTM),
          bOutput4LSTM,
          name = name4Z)

      self.hOutput4LSTM = tf.nn.softmax(self.zOutput4LSTM, name = name4H)
