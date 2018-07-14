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
    self.featureTypes = self.insDatapro.featureTypes
    #self.merged = tf.summary.merge_all()

  # Training and validation for LSTM
  def trainLSTM(self):
    self.xTrainIndex, self.xTestIndex, self.yTrainIndex, self.yTestIndex = \
        self.insDataPro.splitData2TrainAndVal()

    #self.insResultStorer.saveTrainSet(self.xTrainIndex)
    #self.insResultStorer.saveValidationSet(self.xTestIndex)
    #self.insResultStorer.saveTrainLabel(self.yTrainIndex)
    #self.insResultStorer.saveValidationLabel(self.yTestIndex)

    batchSize, bestValAccu = self.FLAGS.batchSize, 0.0
    self.init = tf.global_variables_initializer()

    with tf.Session() as sess:
      saver = tf.train.Saver()
      sess.run(self.init)

      for i in xrange(self.FLAGS.num4Epoches):
        print("No.%d epoch started." %(i))
        ind4xTrainIndex = np.array(range(self.xTrainIndex.shape[0]))
        random.shuffle(ind4xTrainIndex)
        for j in xrange(0, ind4xTrainIndex.shape[0], batchSize):
          # Go through all features
          for featuretype in self.featureTypes:
            


  # Traing and validation for discriminator
  def trainDiscriminator(self):
     self.xTrain, self.xTest, self.yTrain, self.yTest = \
        self.insDataPro.splitData2TrainAndVal()











