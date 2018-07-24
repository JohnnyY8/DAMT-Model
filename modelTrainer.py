# coding = utf-8

import os
import math
import time
import random
import numpy as np
import tensorflow as tf

#from resultStorer import *
from evaluationMetric import *

class ModelTrainer:

  def __init__(self, FLAGS, insDataPro, insCNNModel):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insCNNModel = insCNNModel
    #self.insResultStorer = ResultStorer(FLAGS)
    self.xTrainIndex, self.xTestIndex, self.yTrainIndex, self.yTestIndex = \
        self.insDataPro.splitData2TrainAndVal()

  # Get the dictionary for training CNN model
  def getDict4Train4CNNModel(self, ind4xyTrainIndex, ind4Start):
    batchSize = self.FLAGS.batchSize
    featureTypes = self.insDataPro.featureTypes
    feedDict4Train4CNNModel = {}

    for featureType in featureTypes:
      if featureType == "DrugFingerPrint":
        batchX4DrugFingerPrint = self.insDataPro.data4DrugFingerPrint[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4DrugFingerPrint] = \
            batchX4DrugFingerPrint

      elif featureType == "DrugPhy":
        batchX4DrugPhy = self.insDataPro.data4DrugPhy[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4DrugPhy] = batchX4DrugPhy

      elif featureType == "L1000":
        # A375
        batchX4L1000A375 = self.insDataPro.data4L1000A375[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4L1000A375] = batchX4L1000A375

        # HA1E
        batchX4L1000HA1E = self.insDataPro.data4L1000HA1E[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4L1000HA1E] = batchX4L1000HA1E

        # HT29
        batchX4L1000HT29 = self.insDataPro.data4L1000HT29[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4L1000HT29] = batchX4L1000HT29

        # MCF7
        batchX4L1000MCF7 = self.insDataPro.data4L1000MCF7[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4L1000MCF7] = batchX4L1000MCF7

        # PC3
        batchX4L1000PC3 = self.insDataPro.data4L1000PC3[
            self.xTrainIndex[
                ind4xyTrainIndex[
                    ind4Start: ind4Start + batchSize]]]

        feedDict4Train4CNNModel[self.insCNNModel.xData4L1000PC3] = batchX4L1000PC3

    return feedDict4Train4CNNModel

  # Get the dictionary for testing CNN model
  def getDict4Test4CNNModel(self, ind4Start):
    batchSize = self.FLAGS.batchSize
    featureTypes = self.insDataPro.featureTypes
    feedDict4Test4CNNModel = {}

    for featureType in featureTypes:
      if featureType == "DrugFingerPrint":
        testX4DrugFingerPrint = \
            self.insDataPro.data4DrugFingerPrint[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4DrugFingerPrint] = testX4DrugFingerPrint

      elif featureType == "DrugPhy":
        testX4DrugPhy = \
            self.insDataPro.data4DrugPhy[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4DrugPhy] = testX4DrugPhy

      elif featureType == "L1000":
        # A375
        testX4L1000A375 = \
            self.insDataPro.data4L1000A375[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4L1000A375] = testX4L1000A375

        # HA1E
        testX4L1000HA1E = \
            self.insDataPro.data4L1000HA1E[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4L1000HA1E] = testX4L1000HA1E

        # HT29
        testX4L1000HT29 = \
            self.insDataPro.data4L1000HT29[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4L1000HT29] = testX4L1000HT29

        # MCF7
        testX4L1000MCF7 = \
            self.insDataPro.data4L1000MCF7[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4L1000MCF7] = testX4L1000MCF7

        # PC3
        testX4L1000PC3 = \
            self.insDataPro.data4L1000PC3[ind4Start: ind4Start + batchSize]

        feedDict4Test4CNNModel[self.insCNNModel.xData4L1000PC3] = testX4L1000PC3

    return feedDict4Test4CNNModel

  # Training and validation for LSTM
  def trainLSTM(self, insBiLSTM):
    self.insBiLSTM = insBiLSTM

    #self.insResultStorer.saveTrainSet(self.xTrainIndex)
    #self.insResultStorer.saveValidationSet(self.xTestIndex)
    #self.insResultStorer.saveTrainLabel(self.yTrainIndex)
    #self.insResultStorer.saveValidationLabel(self.yTestIndex)

    batchSize, bestValAccu = self.FLAGS.batchSize, 0.0
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      #saver = tf.train.Saver()
      sess.run(init)

      for i in xrange(self.FLAGS.num4Epoches):
        print("No.%d epoch started." % (i))

        # For training
        ind4xyTrainIndex = np.array(range(self.xTrainIndex.shape[0]))
        random.shuffle(ind4xyTrainIndex)

        for j in xrange(0, ind4xyTrainIndex.shape[0], batchSize):
          feedDict4Train = self.getDict4Train4CNNModel(ind4xyTrainIndex, j)

          batchY4Classification = self.insDataPro.label4Classification[
              self.yTrainIndex[
                  ind4xyTrainIndex[
                      j: j + batchSize]]]

          feedDict4Train[self.insBiLSTM.yLabel4Classification] = batchY4Classification

          loss, _ = sess.run([self.insBiLSTM.loss4Classification, self.insBiLSTM.trainStep], feed_dict = feedDict4Train)
          print("loss:", loss)

        # For validation
        hammingLoss, oneError, coverage, rankingLoss, jaccardIndex, averagePrecision = \
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for j in xrange(0, self.xTestIndex.shape[0], batchSize):
          feedDict4Test = self.getDict4Test4CNNModel(j)

          testY4Classification = self.insDataPro.label4Classification[self.xTestIndex[j: j + batchSize]]
          feedDict4Test[self.insBiLSTM.yLabel4Classification] = testY4Classification

          score = sess.run(self.insBiLSTM.outputH4LSTM, feed_dict = feedDict4Test)

          hammingLoss += EvaluationMetric.getHammingLoss(score, testY4Classification) * batchSize
          oneError += EvaluationMetric.getOneError(score, testY4Classification) * batchSize
          coverage += EvaluationMetric.getCoverage(score, testY4Classification) * batchSize
          rankingLoss += EvaluationMetric.getRankingLoss(score, testY4Classification) * batchSize
          jaccardIndex += EvaluationMetric.getJaccardIndex(score, testY4Classification) * batchSize
          averagePrecision += EvaluationMetric.getAveragePrecision(score, testY4Classification) * batchSize

        print "  hammingLoss:", hammingLoss / self.xTestIndex.shape[0]
        print "  oneError:", oneError / self.xTestIndex.shape[0]
        print "  coverage:", coverage / self.xTestIndex.shape[0]
        print "  rankingLoss:", rankingLoss / self.xTestIndex.shape[0]
        print "  jaccardIndex:", jaccardIndex / self.xTestIndex.shape[0]
        print "  averagePrecision:", averagePrecision / self.xTestIndex.shape[0]

  # Traing and validation for discriminator
  def trainDiscriminator(self, insDiscriminator):
    self.insDiscriminator = insDiscriminator

    batchSize, bestValAccu = self.FLAGS.batchSize, 0.0
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      #saver = tf.train.Saver()
      sess.run(init)

      for i in xrange(self.FLAGS.num4Epoches):
        print("No.%d epoch started." % (i))

        # For training
        ind4xyTrainIndex = np.array(range(self.xTrainIndex.shape[0]))
        random.shuffle(ind4xyTrainIndex)

        for j in xrange(0, ind4xyTrainIndex.shape[0], batchSize):
          feedDict4Train = self.getDict4Train4CNNModel(ind4xyTrainIndex, j)

          feedDict4Train[self.insDiscriminator.yLabel4Discriminator] = \
              self.insDataPro.label4Discriminator

          loss, accu, _ = sess.run(
              [self.insDiscriminator.loss4Discriminator,
               self.insDiscriminator.accu4Discriminator,
               self.insDiscriminator.trainStep],
              feed_dict = feedDict4Train)
        print("loss:", loss)
        print("accu:", accu)


