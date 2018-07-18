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

  # Get a graph for CNN model
  def getCNNModel(self):
    num4Classes = self.FLAGS.num4Classes
    featureTypes = self.insDataPro.featureTypes
    embeddingDimension = self.FLAGS.embeddingDimension
    num4FeatureTypes = self.insDataPro.num4FeatureTypes
    num4Features4Instance = self.insDataPro.num4Features4Instance
    num4Features4L1000 = self.insDataPro.num4Features4L1000
    num4Features4DrugPhy = self.insDataPro.num4Features4DrugPhy
    num4Features4DrugFingerPrint = self.insDataPro.num4Features4DrugFingerPrint

    #self.keepProb = tf.placeholder(
    #    tf.float32,
    #    name = "keepProb")

    # ===== Input layer =====
    with tf.variable_scope("inputLayer"):
      num4InputChannels = self.FLAGS.num4InputChannels4Input

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # xData and xInput for drug finger print
          self.xData4DrugFingerPrint = tf.placeholder(
              tf.float32,
              [None,
               num4Features4DrugFingerPrint],
              name = "xData4DrugFingerPrint")

          self.xInput4DrugFingerPrint = tf.reshape(
              self.xData4DrugFingerPrint,
              [-1,
               1,
               num4Features4DrugFingerPrint,
               num4InputChannels],
              name = "xInput4DrugFingerPrint")

        elif featureType == "DrugPhy":
          # xData and xInput for drug phy
          self.xData4DrugPhy = tf.placeholder(
              tf.float32,
              [None,
               num4Features4DrugPhy],
              name = "xData4DrugPhy")

          self.xInput4DrugPhy = tf.reshape(
              self.xData4DrugPhy,
              [-1,
               1,
               num4Features4DrugPhy,
               num4InputChannels],
              name = "xInput4DrugPhy")

        elif featureType == "L1000":
          # xData and xInput for L1000 A375
          self.xData4L1000A375 = tf.placeholder(
              tf.float32,
              [None,
               num4Features4L1000],
              name = "xData4L1000A375")

          self.xInput4L1000A375 = tf.reshape(
              self.xData4L1000A375,
              [-1,
               1,
               num4Features4L1000,
               num4InputChannels],
              name = "xInput4L1000A375")

          # xData and xInput for L1000 HA1E
          self.xData4L1000HA1E = tf.placeholder(
              tf.float32,
              [None,
               num4Features4L1000],
              name = "xData4L1000HA1E")

          self.xInput4L1000HA1E = tf.reshape(
              self.xData4L1000HA1E,
              [-1,
               1,
               num4Features4L1000,
               num4InputChannels],
              name = "xInput4L1000HA1E")

          # xData and xInput for L1000 HT29
          self.xData4L1000HT29 = tf.placeholder(
              tf.float32,
              [None,
               num4Features4L1000],
              name = "xData4L1000HT29")

          self.xInput4L1000HT29 = tf.reshape(
              self.xData4L1000HT29,
              [-1,
               1,
               num4Features4L1000,
               num4InputChannels],
              name = "xInput4L1000HT29")

          # xData and xInput for L1000 MCF7
          self.xData4L1000MCF7 = tf.placeholder(
              tf.float32,
              [None,
               num4Features4L1000],
              name = "xData4L1000MCF7")

          self.xInput4L1000MCF7 = tf.reshape(
              self.xData4L1000MCF7,
              [-1,
               1,
               num4Features4L1000,
               num4InputChannels],
              name = "xInput4L1000MCF7")

          # xData and xInput for L1000 PC3
          self.xData4L1000PC3 = tf.placeholder(
              tf.float32,
              [None,
               num4Features4L1000],
              name = "xData4L1000PC3")

          self.xInput4L1000PC3 = tf.reshape(
              self.xData4L1000PC3,
              [-1,
               1,
               num4Features4L1000,
               num4InputChannels],
              name = "xInput4L1000PC3")

      # yLabel for discriminator
      self.yLabel4Discriminator = tf.placeholder(
          tf.float32,
          [None, num4FeatureTypes],
          name = "yLabel4Discriminator")

      # yLabel for classification
      self.yLabel4Classification = tf.placeholder(
          tf.float32,
          [None, num4Classes],
          name = "yLabel4Classification")


    # ===== First convolutional layer =====
    with tf.variable_scope("conv1Layer"):
      # W and B are weight and bias for short.
      # Z means before activation, H means activation value.
      name4W, name4B = "conv1W", "conv1B"
      name4Z4DrugFingerPrint, name4H4DrugFingerPrint = \
          "conv1Z4DrungFingerPrint", "conv1H4DrugFingerPrint"
      name4Z4DrugPhy, name4H4DrugPhy = "conv1Z4DrugPhy", "conv1H4DrugPhy"
      name4Z4L1000A375, name4H4L1000A375 = "conv1Z4L1000A375", "conv1H4L1000A375"
      name4Z4L1000HA1E, name4H4L1000HA1E = "conv1Z4L1000HA1E", "conv1H4L1000HA1E"
      name4Z4L1000HT29, name4H4L1000HT29 = "conv1Z4L1000HT29", "conv1H4L1000HT29"
      name4Z4L1000MCF7, name4H4L1000MCF7 = "conv1Z4L1000MCF7", "conv1H4L1000MCF7"
      name4Z4L1000PC3, name4H4L1000PC3 = "conv1Z4L1000PC3", "conv1H4L1000PC3"

      # Initial weight and bias in Conv1
      # Input channels and output channels in Conv1 are 1, 320
      num4InputChannels, num4OutputChannels = \
          self.FLAGS.num4InputChannels4Conv1, \
          self.FLAGS.num4OutputChannels4Conv1

      conv1KHeight, conv1KWidth = 1, 8
      conv1SHeight, conv1SWidth = 1, 1

      conv1W = self.init_weight_variable(
          name4W,
          [conv1KHeight,
           conv1KWidth,
           num4InputChannels,
           num4OutputChannels])

      conv1B = self.init_bias_variable(
          name4B,
          [num4OutputChannels])

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # Action for drug finger print
          conv1Z4DrugFingerPrint = tf.add(
              self.conv2d(
                  self.xInput4DrugFingerPrint,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4DrugFingerPrint)

          self.conv1H4DrugFingerPrint = tf.nn.relu(
              conv1Z4DrugFingerPrint,
              name = name4H4DrugFingerPrint)

        elif featureType == "DrugPhy":
          # Action for drug phy
          conv1Z4DrugPhy = tf.add(
              self.conv2d(
                  self.xInput4DrugPhy,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4DrugPhy)

          self.conv1H4DrugPhy = tf.nn.relu(
              conv1Z4DrugPhy,
              name = name4H4DrugPhy)

        elif featureType == "L1000":
          # Action for L1000 A375
          conv1Z4L1000A375 = tf.add(
              self.conv2d(
                  self.xInput4L1000A375,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4L1000A375)

          self.conv1H4L1000A375 = tf.nn.relu(
              conv1Z4L1000A375,
              name = name4H4L1000A375)

          # Action for L1000 HA1E
          conv1Z4L1000HA1E = tf.add(
              self.conv2d(
                  self.xInput4L1000HA1E,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4L1000HA1E)

          self.conv1H4L1000HA1E = tf.nn.relu(
              conv1Z4L1000HA1E,
              name = name4H4L1000HA1E)

          # Action for L1000 HT29
          conv1Z4L1000HT29 = tf.add(
              self.conv2d(
                  self.xInput4L1000HT29,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4L1000HT29)

          self.conv1H4L1000HT29 = tf.nn.relu(
              conv1Z4L1000HT29,
              name = name4H4L1000HT29)

          # Action for L1000 MCF7
          conv1Z4L1000MCF7 = tf.add(
              self.conv2d(
                  self.xInput4L1000MCF7,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4L1000MCF7)

          self.conv1H4L1000MCF7 = tf.nn.relu(
              conv1Z4L1000MCF7,
              name = name4H4L1000MCF7)

          # Action for L1000 PC3
          conv1Z4L1000PC3 = tf.add(
              self.conv2d(
                  self.xInput4L1000PC3,
                  conv1W,
                  conv1SHeight,
                  conv1SWidth),
              conv1B,
              name = name4Z4L1000PC3)

          self.conv1H4L1000PC3 = tf.nn.relu(
              conv1Z4L1000PC3,
              name = name4H4L1000PC3)


    # ===== First pooling layer =====
    with tf.variable_scope("pooling1Layer"):
      name4H4DrugFingerPring = "pooling1H4DrugFingerPrint"
      name4H4DrugPhy = "pooling1H4DrugPhy"
      name4H4L1000A375 = "pooling1H4L1000A375"
      name4H4L1000HA1E = "pooling1H4L1000HA1E"
      name4H4L1000HT29 = "pooling1H4L1000HT29"
      name4H4L1000MCF7 = "pooling1H4L1000MCF7"
      name4H4L1000PC3 = "pooling1H4L1000PC3"

      pool1KHeight, pool1KWidth = 1, 4
      pool1SHeight, pool1SWidth = 1, 4

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # Action for drug finger print
          self.pooling1H4DrugFingerPrint = self.max_pool(
              self.conv1H4DrugFingerPrint,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4DrugFingerPrint)

        elif featureType == "DrugPhy":
          # Action for drug phy
          self.pooling1H4DrugPhy = self.max_pool(
              self.conv1H4DrugPhy,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4DrugPhy)

        elif featureType == "L1000":
          # Action for L1000 A375
          self.pooling1H4L1000A375 = self.max_pool(
              self.conv1H4L1000A375,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4L1000A375)

          # Action for L1000 HA1E
          self.pooling1H4L1000HA1E = self.max_pool(
              self.conv1H4L1000HA1E,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4L1000HA1E)

          # Action for L1000 HT29
          self.pooling1H4L1000HT29 = self.max_pool(
              self.conv1H4L1000HT29,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4L1000HT29)

          # Action for L1000 MCF7
          self.pooling1H4L1000MCF7 = self.max_pool(
              self.conv1H4L1000MCF7,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4L1000MCF7)

          # Action for L1000 PC3
          self.pooling1H4L1000PC3 = self.max_pool(
              self.conv1H4L1000PC3,
              pool1KHeight,
              pool1KWidth,
              pool1SHeight,
              pool1SWidth,
              name4H4L1000PC3)


    # ===== Second convolutional layer =====
    with tf.variable_scope("conv2Layer"):
      # W and B are weight and bias for short.
      # Z means before activation, H means activation value.
      name4W, name4B = "conv2W", "conv2B"
      name4Z4DrugFingerPrint, name4H4DrugFingerPring = \
          "conv2Z4DrungFingerPrint", "conv2H4DrugFingerPrint"
      name4Z4DrugPhy, name4H4DrugPhy = "conv2Z4DrugPhy", "conv2H4DrugPhy"
      name4Z4L1000A375, name4H4L1000A375 = "conv2Z4L1000A375", "conv2H4L1000A375"
      name4Z4L1000HA1E, name4H4L1000HA1E = "conv2Z4L1000HA1E", "conv2H4L1000HA1E"
      name4Z4L1000HT29, name4H4L1000HT29 = "conv2Z4L1000HT29", "conv2H4L1000HT29"
      name4Z4L1000MCF7, name4H4L1000MCF7 = "conv2Z4L1000MCF7", "conv2H4L1000MCF7"
      name4Z4L1000PC3, name4H4L1000PC3 = "conv2Z4L1000PC3", "conv2H4L1000PC3"

      # Initial weight and bias in Conv2
      # Input channels and output channels in Conv2 are 320, 480
      num4InputChannels, num4OutputChannels = \
          self.FLAGS.num4InputChannels4Conv2, \
          self.FLAGS.num4OutputChannels4Conv2

      conv2KHeight, conv2KWidth = 1, 8
      conv2SHeight, conv2SWidth = 1, 1

      conv2W = self.init_weight_variable(
          name4W,
          [conv2KHeight,
           conv2KWidth,
           num4InputChannels,
           num4OutputChannels])

      conv2B = self.init_bias_variable(
          name4B,
          [num4OutputChannels])

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # Action for drug finger print
          conv2Z4DrugFingerPrint = tf.add(
              self.conv2d(
                  self.pooling1H4DrugFingerPrint,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4DrugFingerPrint)

          self.conv2H4DrugFingerPrint = tf.nn.relu(
              conv2Z4DrugFingerPrint,
              name = name4H4DrugFingerPrint)

        elif featureType == "DrugPhy":
          # Action for drug phy
          conv2Z4DrugPhy = tf.add(
              self.conv2d(
                  self.pooling1H4DrugPhy,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4DrugPhy)

          self.conv2H4DrugPhy = tf.nn.relu(
              conv2Z4DrugPhy,
              name = name4H4DrugPhy)

        elif featureType == "L1000":
          # Action for L1000 A375
          conv2Z4L1000A375 = tf.add(
              self.conv2d(
                  self.pooling1H4L1000A375,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4L1000A375)

          self.conv2H4L1000A375 = tf.nn.relu(
              conv2Z4L1000A375,
              name = name4H4L1000A375)

          # Action for L1000 HA1E
          conv2Z4L1000HA1E = tf.add(
              self.conv2d(
                  self.pooling1H4L1000HA1E,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4L1000HA1E)

          self.conv2H4L1000HA1E = tf.nn.relu(
              conv2Z4L1000HA1E,
              name = name4H4L1000HA1E)

          # Action for L1000 HT29
          conv2Z4L1000HT29 = tf.add(
              self.conv2d(
                  self.pooling1H4L1000HT29,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4L1000HT29)

          self.conv2H4L1000HT29 = tf.nn.relu(
              conv2Z4L1000HT29,
              name = name4H4L1000HT29)

          # Action for L1000 MCF7
          conv2Z4L1000MCF7 = tf.add(
              self.conv2d(
                  self.pooling1H4L1000MCF7,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4L1000MCF7)

          self.conv2H4L1000MCF7 = tf.nn.relu(
              conv2Z4L1000MCF7,
              name = name4H4L1000MCF7)

          # Action for L1000 PC3
          conv2Z4L1000PC3 = tf.add(
              self.conv2d(
                  self.pooling1H4L1000PC3,
                  conv2W,
                  conv2SHeight,
                  conv2SWidth),
              conv2B,
              name = name4Z4L1000PC3)

          self.conv2H4L1000PC3 = tf.nn.relu(
              conv2Z4L1000PC3,
              name = name4H4L1000PC3)


    # ===== Second pooling layer =====
    with tf.variable_scope("pooling2Layer"):
      name4H4DrugFingerPring = "pooling2H4DrugFingerPrint"
      name4H4DrugPhy = "pooling2H4DrugPhy"
      name4H4L1000A375 = "pooling2H4L1000A375"
      name4H4L1000HA1E = "pooling2H4L1000HA1E"
      name4H4L1000HT29 = "pooling2H4L1000HT29"
      name4H4L1000MCF7 = "pooling2H4L1000MCF7"
      name4H4L1000PC3 = "pooling2H4L1000PC3"

      pool2KHeight, pool2KWidth = 1, 4
      pool2SHeight, pool2SWidth = 1, 4

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # Action for drug finger print
          self.pooling2H4DrugFingerPrint = self.max_pool(
              self.conv2H4DrugFingerPrint,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4DrugFingerPrint)

        elif featureType == "DrugPhy":
          # Action for drug phy
          self.pooling2H4DrugPhy = self.max_pool(
              self.conv2H4DrugPhy,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4DrugPhy)

        elif featureType == "L1000":
          # Action for L1000 A375
          self.pooling2H4L1000A375 = self.max_pool(
              self.conv2H4L1000A375,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4L1000A375)

          # Action for L1000 HA1E
          self.pooling2H4L1000HA1E = self.max_pool(
              self.conv2H4L1000HA1E,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4L1000HA1E)

          # Action for L1000 HT29
          self.pooling2H4L1000HT29 = self.max_pool(
              self.conv2H4L1000HT29,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4L1000HT29)

          # Action for L1000 MCF7
          self.pooling2H4L1000MCF7 = self.max_pool(
              self.conv2H4L1000MCF7,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4L1000MCF7)

          # Action for L1000 PC3
          self.pooling2H4L1000PC3 = self.max_pool(
              self.conv2H4L1000PC3,
              pool2KHeight,
              pool2KWidth,
              pool2SHeight,
              pool2SWidth,
              name4H4L1000PC3)


    # ===== Third convolutional layer =====
    with tf.variable_scope("conv3Layer"):
      # W and B are weight and bias for short.
      # Z means before activation, H means activation value.
      name4W, name4B = "conv3W", "conv3B"
      name4Z4DrugFingerPrint, name4H4DrugFingerPring = \
          "conv3Z4DrungFingerPrint", "conv3H4DrugFingerPrint"
      name4Z4DrugPhy, name4H4DrugPhy = "conv3Z4DrugPhy", "conv3H4DrugPhy"
      name4Z4L1000A375, name4H4L1000A375 = "conv3Z4L1000A375", "conv3H4L1000A375"
      name4Z4L1000HA1E, name4H4L1000HA1E = "conv3Z4L1000HA1E", "conv3H4L1000HA1E"
      name4Z4L1000HT29, name4H4L1000HT29 = "conv3Z4L1000HT29", "conv3H4L1000HT29"
      name4Z4L1000MCF7, name4H4L1000MCF7 = "conv3Z4L1000MCF7", "conv3H4L1000MCF7"
      name4Z4L1000PC3, name4H4L1000PC3 = "conv3Z4L1000PC3", "conv3H4L1000PC3"

      # Initial weight and bias in Conv1
      # Input channels and output channels in Conv3 are 480, 960.
      num4InputChannels, num4OutputChannels = \
          self.FLAGS.num4InputChannels4Conv3, \
          self.FLAGS.num4OutputChannels4Conv3

      conv3KHeight, conv3KWidth = 1, 8
      conv3SHeight, conv3SWidth = 1, 1

      conv3W = self.init_weight_variable(
          name4W,
          [conv3KHeight,
           conv3KWidth,
           num4InputChannels,
           num4OutputChannels])

      conv3B = self.init_bias_variable(
          name4B,
          [num4OutputChannels])

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # Action for drug finger print
          conv3Z4DrugFingerPrint = tf.add(
              self.conv2d(
                  self.pooling2H4DrugFingerPrint,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4DrugFingerPrint)

          self.conv3H4DrugFingerPrint = tf.nn.relu(
              conv3Z4DrugFingerPrint,
              name = name4H4DrugFingerPrint)

          self.shape4Conv3H4DrugFingerPrint = \
              self.conv3H4DrugFingerPrint.get_shape().as_list()

        elif featureType == "DrugPhy":
          # Action for drug phy
          conv3Z4DrugPhy = tf.add(
              self.conv2d(
                  self.pooling2H4DrugPhy,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4DrugPhy)

          self.conv3H4DrugPhy = tf.nn.relu(
              conv3Z4DrugPhy,
              name = name4H4DrugPhy)

          self.shape4Conv3H4DrugPhy = \
              self.conv3H4DrugPhy.get_shape().as_list()

        elif featureType == "L1000":
          # Action for L1000 A375
          conv3Z4L1000A375 = tf.add(
              self.conv2d(
                  self.pooling2H4L1000A375,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4L1000A375)

          self.conv3H4L1000A375 = tf.nn.relu(
              conv3Z4L1000A375,
              name = name4H4L1000A375)

          self.shape4Conv3H4L1000A375 = \
              self.conv3H4L1000A375.get_shape().as_list()

          # Action for L1000 HA1E
          conv3Z4L1000HA1E = tf.add(
              self.conv2d(
                  self.pooling2H4L1000HA1E,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4L1000HA1E)

          self.conv3H4L1000HA1E = tf.nn.relu(
              conv3Z4L1000HA1E,
              name = name4H4L1000HA1E)

          self.shape4Conv3H4L1000HA1E = \
              self.conv3H4L1000HA1E.get_shape().as_list()

          # Action for L1000 HT29
          conv3Z4L1000HT29 = tf.add(
              self.conv2d(
                  self.pooling2H4L1000HT29,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4L1000HT29)

          self.conv3H4L1000HT29 = tf.nn.relu(
              conv3Z4L1000HT29,
              name = name4H4L1000HT29)

          self.shape4Conv3H4L1000HT29 = \
              self.conv3H4L1000HT29.get_shape().as_list()

          # Action for L1000 MCF7
          conv3Z4L1000MCF7 = tf.add(
              self.conv2d(
                  self.pooling2H4L1000MCF7,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4L1000MCF7)

          self.conv3H4L1000MCF7 = tf.nn.relu(
              conv3Z4L1000MCF7,
              name = name4H4L1000MCF7)

          self.shape4Conv3H4L1000MCF7 = \
              self.conv3H4L1000MCF7.get_shape().as_list()

          # Action for L1000 PC3
          conv3Z4L1000PC3 = tf.add(
              self.conv2d(
                  self.pooling2H4L1000PC3,
                  conv3W,
                  conv3SHeight,
                  conv3SWidth),
              conv3B,
              name = name4Z4L1000PC3)

          self.conv3H4L1000PC3 = tf.nn.relu(
              conv3Z4L1000PC3,
              name = name4H4L1000PC3)

          self.shape4Conv3H4L1000PC3 = \
              self.conv3H4L1000PC3.get_shape().as_list()


    # ===== Appending layer =====
    with tf.variable_scope("appendingLayer"):
      name4H4DrugFingerPrint = "input4FixedSize4DrugFingerPrint"
      name4H4DrugPhy = "input4FixedSize4DrugPhy"
      name4H4L1000A375 = "input4FixedSize4L1000A375"
      name4H4L1000HA1E = "input4FixedSize4L1000HA1E"
      name4H4L1000HT29 = "input4FixedSize4L1000HT29"
      name4H4L1000MCF7 = "input4FixedSize4L1000MCF7"
      name4H4L1000PC3 = "input4FixedSize4L1000PC3"

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # Length of all feature maps for drug finger print
          len4FeatureMaps4DrugFingerPrint = \
              self.shape4Conv3H4DrugFingerPrint[1] * \
              self.shape4Conv3H4DrugFingerPrint[2] * \
              self.shape4Conv3H4DrugFingerPrint[3]

          self.input4FixedSize4DrugFingerPrint = tf.reshape(
              self.conv3H4DrugFingerPrint,
              [-1, 1, len4FeatureMaps4DrugFingerPrint, 1],
              name = name4H4DrugFingerPrint)

        elif featureType == "DrugPhy":
          # Length of all feature maps for drug phy
          len4FeatureMaps4DrugPhy = \
              self.shape4Conv3H4DrugPhy[1] * \
              self.shape4Conv3H4DrugPhy[2] * \
              self.shape4Conv3H4DrugPhy[3]

          self.input4FixedSize4DrugPhy = tf.reshape(
              self.conv3H4DrugPhy,
              [-1, 1, len4FeatureMaps4DrugPhy, 1],
              name = name4H4DrugPhy)

        elif featureType == "L1000":
          # Length of all feature maps for L1000 A375
          len4FeatureMaps4L1000A375 = \
              self.shape4Conv3H4L1000A375[1] * \
              self.shape4Conv3H4L1000A375[2] * \
              self.shape4Conv3H4L1000A375[3]

          self.input4FixedSize4L1000A375 = tf.reshape(
              self.conv3H4L1000A375,
              [-1, 1, len4FeatureMaps4L1000A375, 1],
              name = name4H4L1000A375)

          # Length of all feature maps for L1000 HA1E
          len4FeatureMaps4L1000HA1E = \
              self.shape4Conv3H4L1000HA1E[1] * \
              self.shape4Conv3H4L1000HA1E[2] * \
              self.shape4Conv3H4L1000HA1E[3]

          self.input4FixedSize4L1000HA1E = tf.reshape(
              self.conv3H4L1000HA1E,
              [-1, 1, len4FeatureMaps4L1000HA1E, 1],
              name = name4H4L1000HA1E)

          # Length of all feature maps for L1000 HT29
          len4FeatureMaps4L1000HT29 = \
              self.shape4Conv3H4L1000HT29[1] * \
              self.shape4Conv3H4L1000HT29[2] * \
              self.shape4Conv3H4L1000HT29[3]

          self.input4FixedSize4L1000HT29 = tf.reshape(
              self.conv3H4L1000HT29,
              [-1, 1, len4FeatureMaps4L1000HT29, 1],
              name = name4H4L1000HT29)

          # Length of all feature maps for L1000 MCF7
          len4FeatureMaps4L1000MCF7 = \
              self.shape4Conv3H4L1000MCF7[1] * \
              self.shape4Conv3H4L1000MCF7[2] * \
              self.shape4Conv3H4L1000MCF7[3]

          self.input4FixedSize4L1000MCF7 = tf.reshape(
              self.conv3H4L1000MCF7,
              [-1, 1, len4FeatureMaps4L1000MCF7, 1],
              name = name4H4L1000MCF7)

          # Length of all feature maps for L1000 PC3
          len4FeatureMaps4L1000PC3 = \
              self.shape4Conv3H4L1000PC3[1] * \
              self.shape4Conv3H4L1000PC3[2] * \
              self.shape4Conv3H4L1000PC3[3]

          self.input4FixedSize4L1000PC3 = tf.reshape(
              self.conv3H4L1000PC3,
              [-1, 1, len4FeatureMaps4L1000PC3, 1],
              name = name4H4L1000PC3)


    # ===== ROI pooling layer =====
    with tf.variable_scope("roiPoolingLayer"):
      name4H4DrugFingerPrint = "roiPoolingH4DrugFingerPrint"
      name4H4DrugPhy = "roiPoolingH4DrugFingerPhy"
      name4H4L1000A375 = "roiPoolingH4L1000A375"
      name4H4L1000HA1E = "roiPoolingH4L1000HA1E"
      name4H4L1000HT29 = "roiPoolingH4L1000HT29"
      name4H4L1000MCF7 = "roiPoolingH4L1000MCF7"
      name4H4L1000PC3 = "roiPoolingH4L1000PC3"

      for featureType in featureTypes:
        if featureType == "DrugFingerPrint":
          # ROI pooling for drug finger print
          roiPoolingKHeight4DrugFingerPrint, roiPoolingKWidth4DrugFingerPrint = \
              1, int(math.ceil(len4FeatureMaps4DrugFingerPrint * 1.0 / embeddingDimension))
          roiPoolingSHeight4DrugFingerPrint, roiPoolingSWidth4DrugFingerPrint = \
              1, int(math.ceil(len4FeatureMaps4DrugFingerPrint * 1.0 / embeddingDimension))

          self.roiPoolingH4DrugFingerPrint = self.avg_pool(
              self.input4FixedSize4DrugFingerPrint,
              roiPoolingKHeight4DrugFingerPrint,
              roiPoolingKWidth4DrugFingerPrint,
              roiPoolingSHeight4DrugFingerPrint,
              roiPoolingSWidth4DrugFingerPrint,
              name4H4DrugFingerPrint)

        elif featureType == "DrugPhy":
          # ROI pooling for drug phy
          roiPoolingKHeight4DrugPhy, roiPoolingKWidth4DrugPhy = \
              1, int(math.ceil(len4FeatureMaps4DrugPhy * 1.0 / embeddingDimension))
          roiPoolingSHeight4DrugPhy, roiPoolingSWidth4DrugPhy = \
              1, int(math.ceil(len4FeatureMaps4DrugPhy * 1.0 / embeddingDimension))

          self.roiPoolingH4DrugPhy = self.avg_pool(
              self.input4FixedSize4DrugPhy,
              roiPoolingKHeight4DrugPhy,
              roiPoolingKWidth4DrugPhy,
              roiPoolingSHeight4DrugPhy,
              roiPoolingSWidth4DrugPhy,
              name4H4DrugPhy)

        elif featureType == "L1000":
          # ROI pooling for L1000 A375
          roiPoolingKHeight4L1000A375, roiPoolingKWidth4L1000A375 = \
              1, int(math.ceil(len4FeatureMaps4L1000A375 * 1.0 / embeddingDimension))
          roiPoolingSHeight4L1000A375, roiPoolingSWidth4L1000A375 = \
              1, int(math.ceil(len4FeatureMaps4L1000A375 * 1.0 / embeddingDimension))

          self.roiPoolingH4L1000A375 = self.avg_pool(
              self.input4FixedSize4L1000A375,
              roiPoolingKHeight4L1000A375,
              roiPoolingKWidth4L1000A375,
              roiPoolingSHeight4L1000A375,
              roiPoolingSWidth4L1000A375,
              name4H4L1000A375)

          # ROI pooling for L1000 HA1E
          roiPoolingKHeight4L1000HA1E, roiPoolingKWidth4L1000HA1E = \
              1, int(math.ceil(len4FeatureMaps4L1000HA1E * 1.0 / embeddingDimension))
          roiPoolingSHeight4L1000HA1E, roiPoolingSWidth4L1000HA1E = \
              1, int(math.ceil(len4FeatureMaps4L1000HA1E * 1.0 / embeddingDimension))

          self.roiPoolingH4L1000HA1E = self.avg_pool(
              self.input4FixedSize4L1000HA1E,
              roiPoolingKHeight4L1000HA1E,
              roiPoolingKWidth4L1000HA1E,
              roiPoolingSHeight4L1000HA1E,
              roiPoolingSWidth4L1000HA1E,
              name4H4L1000HA1E)

          # ROI pooling for L1000 HT29
          roiPoolingKHeight4L1000HT29, roiPoolingKWidth4L1000HT29 = \
              1, int(math.ceil(len4FeatureMaps4L1000HT29 * 1.0 / embeddingDimension))
          roiPoolingSHeight4L1000HT29, roiPoolingSWidth4L1000HT29 = \
              1, int(math.ceil(len4FeatureMaps4L1000HT29 * 1.0 / embeddingDimension))

          self.roiPoolingH4L1000HT29 = self.avg_pool(
              self.input4FixedSize4L1000HT29,
              roiPoolingKHeight4L1000HT29,
              roiPoolingKWidth4L1000HT29,
              roiPoolingSHeight4L1000HT29,
              roiPoolingSWidth4L1000HT29,
              name4H4L1000HT29)

          # ROI pooling for L1000 MCF7
          roiPoolingKHeight4L1000MCF7, roiPoolingKWidth4L1000MCF7 = \
              1, int(math.ceil(len4FeatureMaps4L1000MCF7 * 1.0 / embeddingDimension))
          roiPoolingSHeight4L1000MCF7, roiPoolingSWidth4L1000MCF7 = \
              1, int(math.ceil(len4FeatureMaps4L1000MCF7 * 1.0 / embeddingDimension))

          self.roiPoolingH4L1000MCF7 = self.avg_pool(
              self.input4FixedSize4L1000MCF7,
              roiPoolingKHeight4L1000MCF7,
              roiPoolingKWidth4L1000MCF7,
              roiPoolingSHeight4L1000MCF7,
              roiPoolingSWidth4L1000MCF7,
              name4H4L1000MCF7)

          # ROI pooling for L1000 PC3
          roiPoolingKHeight4L1000PC3, roiPoolingKWidth4L1000PC3 = \
              1, int(math.ceil(len4FeatureMaps4L1000PC3 * 1.0 / embeddingDimension))
          roiPoolingSHeight4L1000PC3, roiPoolingSWidth4L1000PC3 = \
              1, int(math.ceil(len4FeatureMaps4L1000PC3 * 1.0 / embeddingDimension))

          self.roiPoolingH4L1000PC3 = self.avg_pool(
              self.input4FixedSize4L1000PC3,
              roiPoolingKHeight4L1000PC3,
              roiPoolingKWidth4L1000PC3,
              roiPoolingSHeight4L1000PC3,
              roiPoolingSWidth4L1000PC3,
              name4H4L1000PC3)


    # ===== Concation layer =====
    with tf.variable_scope("concationLayer"):
      for ind, featureType in enumerate(featureTypes):
        if featureType == "DrugFingerPrint":
          if ind == 0:
            self.output4FixedSize = self.roiPoolingH4DrugFingerPrint
          else:
            self.output4FixedSize = tf.concat(
                (self.output4FixedSize, self.roiPoolingH4DrugFingerPrint),
                axis = 2)

        elif featureType == "DrugPhy":
          if ind == 0:
            self.output4FixedSize = self.roiPoolingH4DrugPhy
          else:
            self.output4FixedSize = tf.concat(
                (self.output4FixedSize, self.roiPoolingH4DrugPhy),
                axis = 2)

        elif featureType == "L1000":
          if ind == 0:
            self.output4FixedSize = self.roiPoolingH4L1000A375
          else:
            self.output4FixedSize = tf.concat(
                (self.output4FixedSize, self.roiPoolingH4L1000A375),
                axis = 2)

          self.output4FixedSize = tf.concat(
              (self.output4FixedSize, self.roiPoolingH4L1000HA1E),
              axis = 2)

          self.output4FixedSize = tf.concat(
              (self.output4FixedSize, self.roiPoolingH4L1000HT29),
              axis = 2)

          self.output4FixedSize = tf.concat(
              (self.output4FixedSize, self.roiPoolingH4L1000MCF7),
              axis = 2)

          self.output4FixedSize = tf.concat(
              (self.output4FixedSize, self.roiPoolingH4L1000PC3),
              axis = 2)

      # Reshape output as input for discriminator
      self.output4FixedSize4Discriminator = tf.reshape(
          self.output4FixedSize,
          [-1, num4Features4Instance * embeddingDimension])

      # Transfer output4FixedSize
      #     from [batchSzie, 1, all features, 1] to [batchSize, timeStep, num4Input]
      self.output4FixedSize = tf.reshape(
          self.output4FixedSize,
          [-1, num4Features4Instance, embeddingDimension])

      # Unstack output as input for LSTM
      self.output4FixedSize4LSTM = tf.unstack(
          self.output4FixedSize,
          num4Features4Instance,
          axis = 1)
