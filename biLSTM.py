# coding=utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *
from tensorflow.contrib import rnn

class BiLSTM(CommonModelFunc):

  def __init__(self, FLAGS, insDataPro, insCNNModel):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insCNNModel = insCNNModel

  # Get a graph for bidirectional LSTM
  def getBiLSTM(self):
    # Network Parameters
    num4Input = self.FLAGS.embeddingDimension  # Same as the shape of output from hROIPooling
    timeStep = self.insDataPro.num4Features4Instance  # Same as number of features for each instance
    num4HiddenUnits4LSTM = self.FLAGS.num4HiddenUnits4LSTM  # The dimensionality of hidden output
    num4Classes = self.FLAGS.num4Classes  # The number of ATC classes

    # ===== LSTM layer =====
    with tf.variable_scope("lstmLayer"):
      name4W, name4B = "output4LSTMW", "output4LSTMB"
      name4Z, name4H = "output4LSTMZ", "output4LSTMH"

      outputW4LSTM = self.init_weight_variable(
          name4W,
          [2 * num4HiddenUnits4LSTM,
           num4Classes])

      outputB4LSTM = self.init_bias_variable(
          name4B,
          [num4Classes])

      ###################################
      # Define lstm cells with tensorflow
      # Forward direction cell
      # Backward direction cell
      ###################################
      lstmFwCell = rnn.BasicLSTMCell(num4HiddenUnits4LSTM, forget_bias = 1.0)
      lstmBwCell = rnn.BasicLSTMCell(num4HiddenUnits4LSTM, forget_bias = 1.0)

      # Get lstm cell output
      try:
        self.hiddenOutputs, _, _ = rnn.static_bidirectional_rnn(
            lstmFwCell,
            lstmBwCell,
            self.insCNNModel.output4FixedSize4LSTM,
            dtype = tf.float32)
      except Exception:  # Old TensorFlow version only returns outputs not states
        self.hiddenOutputs = rnn.static_bidirectional_rnn(
            lstmFwCell,
            lstmBwCell,
            self.insCNNModel.output4FixedSize4LSTM,
            dtype = tf.float32)

      self.outputZ4LSTM = tf.add(
          tf.matmul(
              self.hiddenOutputs[-1],
              outputW4LSTM),
          outputB4LSTM,
          name = name4Z)

      self.outputH4LSTM = tf.nn.sigmoid(self.outputZ4LSTM, name = name4H)
      #self.outputH4LSTM = tf.nn.softmax(self.zOutput4LSTM, name = name4H)


    # ===== Loss layer for LSTM =====
    with tf.variable_scope("loss4ClassificationLayer"):
      name4Loss4Classification = "loss4Classification"

      self.loss4Classification = tf.reduce_mean(
          tf.square(
              tf.subtract(
                  self.outputH4LSTM,
                  self.insCNNModel.yLabel4Classification)),
          name = name4Loss4Classification)
      #self.loss4LSTM = tf.reduce_mean(
      #    tf.nn.softmax_cross_entropy_with_logits(
      #        logits = self.zDiscriminator,
      #        labels = self.CNNModel.fType),
      #    name = name4Loss)
      #tf.summary.scalar("loss4LSTM", self.loss4LSTM)

    self.trainStep = tf.train.AdamOptimizer(
        self.FLAGS.learningRate).minimize(self.loss4Classification)
