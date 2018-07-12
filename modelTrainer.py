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

    bestValAccu = 0, 0.0
    batchSize = self.FLAGS.batchSize

    with tf.Session() as sess:
      saver = tf.train.Saver()
      sess.run(self.init)

      for i in xrange(self.FLAGS.num4Epoches):
        print("No.%d epoch started." %(i))
        ind4xTrainIndex = np.array(range(self.xTrainIndex.shape[0]))
        random.shuffle(ind4xTrainIndex)
        for j in xrange(0, ind4xTrainIndex.shape[0], batchSize):
         


  # Traing and validation for discriminator
  def trainDiscriminator(self):
     self.xTrain, self.xTest, self.yTrain, self.yTest = \
        self.insDataPro.splitData2TrainAndVal()











