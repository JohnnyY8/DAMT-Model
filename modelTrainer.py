# coding = utf-8

import os
import math
import time
import random
import numpy as np
import tensorflow as tf

#from resultStorer import *

class ModelTrainer:

  def __init__(self, FLAGS, insDataPro, insCNNModel, insBiLSTM):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insCNNModel = insCNNModel
    self.insBiLSTM = insBiLSTM
    #self.insResultStorer = ResultStorer(FLAGS)
    self.featureTypes = self.insDataPro.featureTypes
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
      #saver = tf.train.Saver()
      sess.run(self.init)

      for i in xrange(self.FLAGS.num4Epoches):
        print("No.%d epoch started." %(i))
        ind4xTrainIndex = np.array(range(self.xTrainIndex.shape[0]))
        ind4yTrainIndex = np.array(range(self.yTrainIndex.shape[0]))
        random.shuffle(ind4xTrainIndex)
        for j in xrange(0, ind4xTrainIndex.shape[0], batchSize):
          batchX4DrugFingerPrint, batchY4DrugFingerPrint = \
              self.insDataPro.data4DrugFingerPrint[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.yTrainIndex[ind4yTrainIndex[j: j + batchSize]]]

          batchX4DrugPhy, batchY4DrugPhy = \
              self.insDataPro.data4DrugPhy[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.yTrainIndex[ind4yTrainIndex[j: j + batchSize]]]

          batchX4L1000A375, batchY4DrugL1000A375 = \
              self.insDataPro.data4L1000A375[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.yTrainIndex[ind4yTrainIndex[j: j + batchSize]]]

          batchX4L1000HA1E, batchY4L1000HA1E = \
              self.insDataPro.data4L1000HA1E[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]]

          batchX4L1000HT29, batchY4L1000HT29 = \
              self.insDataPro.data4L1000HT29[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]]

          batchX4L1000MCF7, batchY4L1000MCF7 = \
              self.insDataPro.data4L1000MCF7[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]]

          batchX4l1000PC3, batchY4L1000PC3 = \
              self.insDataPro.data4L1000PC3[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]], \
              self.insDataPro.label4Classification[self.xTrainIndex[ind4xTrainIndex[j: j + batchSize]]]

          feedDict = { \
              self.insCNNModel.xData4DrugFingerPrint: batchX4DrugFingerPrint, \
              self.insCNNModel.xData4DrugPhy: batchX4DrugPhy, \
              self.insCNNModel.xData4L1000A375: batchX4L1000A375, \
              self.insCNNModel.xData4L1000HA1E: batchX4L1000HA1E, \
              self.insCNNModel.xData4L1000HT29: batchX4L1000HT29, \
              self.insCNNModel.xData4L1000MCF7: batchX4L1000MCF7, \
              self.insCNNModel.xData4L1000PC3: batchX4L1000PC3, \
              self.insCNNModel.yLabel4Discriminator: np.zeros[[2]], \
              self.insCNNModel.yLabel4Classification: batchY4DrugPhy}

          sess.run(self.insBiLSTM.trainStep, feed_dict = feedDict)


















  # Traing and validation for discriminator
  def trainDiscriminator(self):
     self.xTrain, self.xTest, self.yTrain, self.yTest = \
        self.insDataPro.splitData2TrainAndVal()











