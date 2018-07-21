# coding=utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class Discriminator(CommonModelFunc):

  def __init__(self, FLAGS, insDataPro, insCNNModel):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insCNNModel = insCNNModel

  # Get a graph for discriminator
  def getDiscriminator(self):
    embeddingDimension = self.FLAGS.embeddingDimension
    featureTypes = self.insDataPro.featureTypes
    num4FeatureTypes = self.insDataPro.num4FeatureTypes

    # ===== Discriminator layer =====
    with tf.variable_scope("discriminatorLayer"):
      self.yLabel4Discriminator = tf.placeholder(
          tf.float32,
          [None, num4FeatureTypes],
          name = "yLabel4Discriminator")
#      for featureType in featureTypes:
#        if featureType == "DrugFingerPrint":
#          # yLabel of discriminator for drug finger print
#          self.yLabel4Discriminator4DrugFingerPrint = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4DrugFingerPrint")
#
#        elif featureType == "DrugPhy":
#          # yLabel of discriminator for drug phy
#          self.yLabel4Discriminator4DrugPhy = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4DrugPhy")
#
#        elif featureType == "L1000":
#          # yLabel of discriminator for L1000
#          self.yLabel4Discriminator4L1000 = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4L1000")

#          # yLabel of discriminator for L1000 A375
#          self.yLabel4Discriminator4L1000A375 = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4L1000A375")
#
#          # yLabel of discriminator for L1000 HA1E
#          self.yLabel4Discriminator4L1000HA1E = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4L1000HA1E")
#
#          # yLabel of discriminator for L1000 HT29
#          self.yLabel4Discriminator4L1000HT29 = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4L1000HT29")
#
#          # yLabel of discriminator for L1000 MCF7
#          self.yLabel4Discriminator4L1000MCF7 = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4L1000MCF7")
#
#          # yLabel of discriminator for L1000 PC3
#          self.yLabel4Discriminator4L1000PC3 = tf.placeholder(
#              tf.float32,
#              [None, num4FeatureTypes],
#              name = "yLabel4Discriminator4L1000PC3")

      name4W, name4B = "discriminatorW", "iscriminatorB"
      name4Z, name4H = "discriminatorZ", "discriminatorH"

      discriminatorW = self.init_weight_variable(
          name4W,
          [embeddingDimension,
           num4FeatureTypes])

      discriminatorB = self.init_bias_variable(
          name4B,
          [num4FeatureTypes])

      self.discriminatorZ = tf.add(
          tf.matmul(
              self.insCNNModel.output4FixedSize4Discriminator,
              discriminatorW),
          discriminatorB,
          name = name4Z)

      self.discriminatorH = tf.nn.softmax(
          self.discriminatorZ,
          name = name4H)

    # ===== Loss layer for discriminator =====
    with tf.variable_scope("loss4DiscriminatorLayer"):
      name4Loss = "loss4Discriminator"

      self.loss4Discriminator = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits = self.discriminatorZ,
              labels = self.yLabel4Discriminator),
          name = name4Loss)
      #tf.summary.scalar("loss4Discriminator", tf.reduce_mean(self.loss4Discriminator))

      self.trainStep = tf.train.AdamOptimizer(
          self.FLAGS.learningRate).minimize(self.loss4Discriminator)

    # ===== Accuracy layer for discriminator =====
    with tf.variable_scope("accu4DiscriminatorLayer"):
      name4Accu = "accu4Discriminator"

      correctPrediction = tf.equal(
          tf.argmax(self.discriminatorH, 1),
          tf.argmax(self.yLabel4Discriminator, 1))

      self.accu4Discriminator = tf.reduce_mean(
          tf.cast(correctPrediction, tf.float32),
          name = name4Accu)
      #tf.summary.scalar("accu4Discriminator", self.accu4Discriminator)


