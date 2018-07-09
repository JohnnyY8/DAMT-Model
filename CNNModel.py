#coding=utf-8

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
    num4Features = self.FLAGS.num4Features
    num4FeatureTypes = self.FLAGS.num4FeatureTypes
    num4Classes = self.FLAGS.num4Classes

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
      len4AllFM = self.shape4hConv3[2] * self.shape4hConv3[3]
      self.input4FixedSize = tf.reshape(
          self.hConv3,
          [-1, 1, len4AllFM, 1],
          name = name4H)

    # ===== ROI pooling layer =====
    with tf.variable_scope("roiPoolingLayer"):
      name4H = "hROIPooling"

      roiPool1KHeight, roiPool1KWidth = 1, int(math.ceil(len4AllFM * 1.0 / num4FirstFC))  # TODO
      roiPool1SHeight, roiPool1SWidth = 1, int(math.ceil(len4AllFM * 1.0 / num4FirstFC))  # TODO

      # The dimensionality of hROIPooling is 925
      self.hROIPooling = self.max_pool(
          self.input4FixedSize,
          pool1KHeight,
          pool1KWidth,
          pool1SHeight,
          pool1SWidth,
          name = name4H)

      self.shape4hROIPooling = self.hROIPooling.get_shape().as_list()






    # ===== First fully connected layer =====
    with tf.variable_scope("fc1Layer"):
      name4W, name4B = "wFC1", "bFC1"
      name4Z, name4H = "zFC1", "hFC1"

      shape4hROIPooling = hROIPooling.get_shape().as_list()
      len4One = shape4hROIPooling[2]
      input4FC1 = tf.reshape(hROIPooling, [-1, len4One])

      wFC1 = self.init_weight_variable(
          name4Weight,
          [len4One,
           num4FirstFC])
      self.variable_summaries(wFC1)

      bFC1 = self.init_bias_variable(
          name4Bias,
          [num4FirstFC])
      self.variable_summaries(bFC1)

      preActFC1 = tf.add(
          tf.matmul(
              input4FC1,
              wFC1),
          bFC1,
          name = name4PreAct)
      self.variable_summaries(preActFC1)

      hFC1 = tf.nn.relu(
          preActFC1,
          name = name4Act)
      self.variable_summaries(hFC1)

    # Second fully connected layer
    with tf.variable_scope("fc2Layer"):
      name4Weight, name4Bias = "wFC2", "bFC2"
      name4PreAct, name4Act = "preActFC2", "hFC2"

      wFC2 = self.init_weight_variable(
          name4Weight,
          [num4FirstFC, num4SecondFC])
      self.variable_summaries(wFC2)

      bFC2 = self.init_bias_variable(
          name4Bias,
          [num4SecondFC])
      self.variable_summaries(bFC2)

      self.preActFC2 = tf.add(
          tf.matmul(
              hFC1,
              wFC2),
          bFC2,
          name = name4PreAct)
      self.variable_summaries(self.preActFC2)

      self.hFC2 = tf.nn.relu(
          self.preActFC2,
          name = name4Act)
      self.variable_summaries(self.hFC2)

#      self.hFC2DropOut = tf.nn.dropout(
#          hFC2,
#          self.keepProb)
#      self.variable_summaries(hFC2DropOut)

    with tf.variable_scope("outputLayer"):
      name4Weight, name4Bias, name4Act = "wOutput", "bOutput", "hOutput"

      wOutput = self.init_weight_variable(name4Weight, [num4SecondFC, 2])
      bOutput = self.init_bias_variable(name4Bias, [2])
      self.preActOutput = tf.add(tf.matmul(self.hFC2, wOutput), bOutput)
      self.hOutput = tf.nn.softmax(self.preActOutput, name = name4Act)

    # Cost function
    with tf.variable_scope("lossLayer"):
      predPro4PandN = tf.reshape(
          tf.reduce_sum(
              self.hOutput,
              reduction_indices = [0]),
          [-1, 2])

      predPro4P = tf.matmul(
          predPro4PandN,
          tf.constant([[0.], [1.]]))

      predPro4N = tf.matmul(
          predPro4PandN,
          tf.constant([[1.], [0.]]))

      predPro4PandNwithLabel = tf.reshape(
          tf.reduce_mean(
              self.yLabel * self.hOutput,
              reduction_indices = [0]),
          [-1, 2])

      predPro4PwithLabel = tf.matmul(
          predPro4PandNwithLabel,
          tf.constant([[0.], [1.]]))

      predPro4NwithLabel = tf.matmul(
          predPro4PandNwithLabel,
          tf.constant([[1.], [0.]]))

      self.loss = tf.subtract(
          tf.reduce_mean(
              tf.nn.softmax_cross_entropy_with_logits(
                  logits = self.preActOutput,
                  labels = self.yLabel)),
          self.FLAGS.nWeight * predPro4NwithLabel,
          name = "loss")
      tf.summary.scalar("lossValue", tf.reduce_mean(self.loss))

      self.trainStep = tf.train.AdamOptimizer(
          self.FLAGS.learningRate).minimize(self.loss)

    # Accuracy
    with tf.variable_scope("accuracyLayer"):
      correctPrediction = tf.equal(tf.argmax(self.hOutput, 1), tf.argmax(self.yLabel, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
      tf.summary.scalar("accuracy", self.accuracy)

    self.merged = tf.summary.merge_all()

    self.init = tf.global_variables_initializer()
