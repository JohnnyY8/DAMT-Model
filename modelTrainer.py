#coding = utf-8

import os
import math
import time
import random
import numpy as np
import tensorflow as tf

from resultStorer import *

class ModelTrainer:

  def __init__(self, FLAGS, insDataPro, insModel):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insModel = insModel
    self.insResultStorer = ResultStorer(FLAGS)
    #self.merged = tf.summary.merge_all()
    self.init = tf.global_variables_initializer()

  # Training and validation for LSTM
  def trainLSTM(self):
    self.xTrainIndex, self.xTestIndex, self.yTrain, self.yTest = \
        self.insDataPro.splitData2TrainAndVal()

    self.insResultStorer.saveTrainSet(self.xTrainIndex)
    self.insResultStorer.saveValidationSet(self.xTestIndex)
    self.insResultStorer.saveTrainLabel(self.yTrain)
    self.insResultStorer.saveValidationLabel(self.yTest)

    with tf.Session() as sess:
      oldTrainAccu, newTrainAccu, bestValAccu = 0.0, 0.0, 0.0
      flag, num4Epoches = 0, 0

    saver = tf.train.Saver()
    sess.run(self.init)


  # Traing and validation for discriminator
  def trainDiscriminator(self):
     self.xTrain, self.xTest, self.yTrain, self.yTest = \
        self.insDataPro.splitData2TrainAndVal()











