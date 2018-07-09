#coding=utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class Discriminator(CommonModelFunc):

  def __init__(self, FLAGS, insCNNModel, insDataPro):
    self.FLAGS = FLAGS
    self.insCNNModel = insCNNModel
    self.insDataPro = insDataPro  # TODO

  # Get a graph for discriminator
  def getDiscriminator(self):
    num4FeatureTypes = self.FLAGS.num4FeatureTypes

    # ===== Discriminator layer =====
    with tf.variable_scope("discriminatorLayer"):
      name4W, name4B = "wDiscriminator", "bDiscriminator"
      name4Z, name4H = "zDiscriminator", "hDiscriminator"

      wDiscriminator = self.init_weight_variable(
          name4W,
          [self.insCNNModel.shape4hROIPooling,
           num4FeatureTypes])

      bDiscriminator = self.init_bias_variable(
          name4B,
          [num4FeatureTypes])

      self.zDiscriminator = tf.add(
          tf.matmul(
              self.insCNNModel.hROIPooling,
              wDiscriminator),
          bDiscriminator,
          name = name4Z)

      self.hDiscriminator = tf.nn.softmax(
          self.zDiscriminator,
          name = name4H)

    # ===== Loss layer for discriminator =====
    with tf.variable_scope("loss4DiscriminatorLayer"):
      name4Loss = "loss4Discriminator"

      self.loss4Discriminator = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits = self.zDiscriminator,
              labels = self.CNNModel.fType),
          name = name4Loss)
      tf.summary.scalar("loss4Discriminator", tf.reduce_mean(self.loss4Discriminator))

      self.trainStep = tf.train.AdamOptimizer(
          self.FLAGS.learningRate).minimize(self.loss4Discriminator)

    # ===== Accuracy layer for discriminator =====
    with tf.variable_scope("accu4DiscriminatorLayer"):
      name4Accu = "accu4Discriminator"

      correctPrediction = tf.equal(tf.argmax(self.hOutput, 1), tf.argmax(self.yLabel, 1))
      self.accu4Discriminator = tf.reduce_mean(tf.cast(correctPrediction, tf.float32), name = name4Accu)
      tf.summary.scalar("accu4Discriminator", self.accu4Discriminator)

    self.merged = tf.summary.merge_all()

    self.init = tf.global_variables_initializer()
