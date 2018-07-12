# coding = utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class CNNModel(CommonModelFunc):

  def __init__(self, FLAGS, insDataPro):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro

  # Get a graph for base CNN model
  def getCNNModel(self):
    num4Classes = self.FLAGS.num4Classes
    num4FeatureTypes = self.insDataPro.num4FeatureTypes
    num4Features4Instance = self.insDataPro.num4Features4Instance

    self.keepProb = tf.placeholder(
        tf.float32,
        name = "keepProb")

    # ===== Input layer =====
    with tf.variable_scope("inputLayer"):
      self.xData = tf.placeholder(
          tf.float32,
          [None,
           num4Features,
           num4InputChannels],
          name = "xData")

      self.xInput = tf.reshape(
          self.xData,
          [-1, 1, num4Features, num4InputChannels],
          name = "xInput")

      self.fType = tf.placeholder(
          tf.float32,
          [None, num4FeatureTypes],
          name = "fType")

      self.yLabel = tf.placeholder(
          tf.float32,
          [None, num4Classes],
          name = "yLabel")

    # ===== First convolutional layer =====
    with tf.variable_scope("conv1Layer"):
      # W and B are weight and bias for short.
      # Z means before activation, H means activation value.
      name4W, name4B = "wConv1", "bConv1"
      name4Z, name4H = "zConv1", "hConv1"

      # IC and OC are input channels and output channels for short.
      # IC and OC in Conv1 are 1, 320.
      num4IC, num4OC = self.FLAGS.num4IC4Conv1, self.FLAGS.num4OC4Conv1

      conv1KHeight, conv1KWidth = 1, 8
      conv1SHeight, conv1SWidth = 1, 1

      wConv1 = self.init_weight_variable(
          name4W,
          [conv1KHeight,
           conv1KWidth,
           num4IC,
           num4OC])

      bConv1 = self.init_bias_variable(
          name4B,
          [num4OC])

      zConv1 = tf.add(
          self.conv2d(
              self.xInput,
              wConv1,
              conv1SHeight,
              conv1SWidth),
          bConv1,
          name = name4Z)

      self.hConv1 = tf.nn.relu(zConv1, name = name4H)

    # ===== First pooling layer =====
    with tf.variable_scope("pooling1Layer"):
      name4H = "hPooling1"

      pool1KHeight, pool1KWidth = 1, 4
      pool1SHeight, pool1SWidth = 1, 4

      self.hPooling1 = self.max_pool(
          self.hConv1,
          pool1KHeight,
          pool1KWidth,
          pool1SHeight,
          pool1SWidth,
          name = name4H)

    # ===== Second convolutional layer =====
    with tf.variable_scope("conv2Layer"):
      # W and B are weight and bias for short.
      # Z means before activation, H means activation value.
      name4W, name4B = "wConv2", "bConv2"
      name4Z, name4H = "zConv2", "hConv2"

      # IC and OC are input channels and output channels for short.
      # IC and OC in Conv2 are 320, 480.
      num4IC, num4OC = self.FLAGS.num4IC4Conv2, self.FLAGS.num4OC4Conv2

      conv2KHeight, conv2KWidth = 1, 8
      conv2SHeight, conv2SWidth = 1, 1

      wConv2 = self.init_weight_variable(
          name4W,
          [conv2KHeight,
           conv2KWidth,
           num4IC,
           num4OC])

      bConv2 = self.init_bias_variable(
          name4B,
          [num4OC])

      zConv2 = tf.add(
          self.conv2d(
              self.hPooling1,
              wConv2,
              conv2SHeight,
              conv2SWidth),
          bConv2,
          name = name4Z)

      self.hConv2 = tf.nn.relu(zConv2, name = name4H)

    # ===== Second pooling layer =====
    with tf.variable_scope("pooling2Layer"):
      name4H = "hPooling2"

      pool2KHeight, pool2KWidth = 1, 4
      pool2SHeight, pool2SWidth = 1, 4

      self.hPooling2 = self.max_pool(
          self.hConv2,
          pool2KHeight,
          pool2KWidth,
          pool2SHeight,
          pool2SWidth,
          name = name4H)

    # ===== Third convolutional layer =====
    with tf.variable_scope("conv3Layer"):
      # W and B are weight and bias for short.
      # Z means before activation, H means activation value.
      name4W, name4B = "wConv3", "bConv3"
      name4Z, name4H = "zConv3", "hConv3"

      # IC and OC are input channels and output channels for short.
      # IC and OC in Conv2 are 480, 960.
      num4IC, num4OC = self.FLAGS.num4IC4Conv3, self.FLAGS.num4OC4Conv3

      conv3KHeight, conv3KWidth = 1, 8
      conv3SHeight, conv3SWidth = 1, 1

      wConv3 = self.init_weight_variable(
          name4W,
          [conv3KHeight,
           conv3KWidth,
           num4IC,
           num4OC])

      bConv3 = self.init_bias_variable(
          name4B,
          [num4OC])

      zConv3 = tf.add(
          self.conv2d(
              self.hPooling2,
              wConv3,
              conv3SHeight,
              conv3SWidth),
          bConv3,
          name = name4Z)

      self.hConv3 = tf.nn.relu(zConv3, name = name4H)

      self.shape4hConv3 = self.hConv3.get_shape().as_list()

    # ===== Appending layer =====
    with tf.variable_scope("appendingLayer"):
      name4H = "input4FixedSize"

      # FM means feature map for short.
      len4AllFM = self.shape4hConv3[1] * self.shape4hConv3[2] * self.shape4hConv3[3]  # TODO check the shape4hConv3[1]
      self.input4FixedSize = tf.reshape(
          self.hConv3,
          [-1, 1, len4AllFM, 1],
          name = name4H)

    # ===== ROI pooling layer =====
    with tf.variable_scope("roiPoolingLayer"):
      name4H = "hROIPooling"

      roiPool1KHeight, roiPool1KWidth = 1, int(math.ceil(len4AllFM * 1.0 / num4FirstFC))  # TODO
      roiPool1SHeight, roiPool1SWidth = 1, int(math.ceil(len4AllFM * 1.0 / num4FirstFC))  # TODO

      # The dimensionality of hROIPooling is 925 TODO
      self.hROIPooling = self.max_pool(
          self.input4FixedSize,
          pool1KHeight,
          pool1KWidth,
          pool1SHeight,
          pool1SWidth,
          name = name4H)

      self.shape4hROIPooling = self.hROIPooling.get_shape().as_list()

    # ===== First fully connected layer =====  # TODO
    with tf.variable_scope("fc1Layer"):
      name4W, name4B = "wFC1", "bFC1"
      name4Z, name4H = "zFC1", "hFC1"

      input4FC1 = tf.reshape(self.input4FixedSize, [-1, len4AllFM])

      wFC1 = self.init_weight_variable(
          name4W,
          [len4AllFM,
           num4FC1])
      #self.variable_summaries(wFC1)

      bFC1 = self.init_bias_variable(
          name4B,
          [num4FC1])
      #self.variable_summaries(bFC1)

      zFC1 = tf.add(
          tf.matmul(
              input4FC1,
              wFC1),
          bFC1,
          name = name4Z)
      #self.variable_summaries(zFC1)

      self.hFC1 = tf.nn.relu(
          zFC1,
          name = name4H)
      #self.variable_summaries(self.hFC1)

