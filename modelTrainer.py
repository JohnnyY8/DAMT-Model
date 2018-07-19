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
    #self.merged = tf.summary.merge_all()

  # Training and validation for LSTM
  def trainLSTM(self, insBiLSTM):
    self.insBiLSTM = insBiLSTM
    featureTypes = self.insDataPro.featureTypes

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

        feedDict4Test = {}
        ind4xyTrainIndex = np.array(range(self.xTrainIndex.shape[0]))
        random.shuffle(ind4xyTrainIndex)

        for j in xrange(0, ind4xyTrainIndex.shape[0], batchSize):
          feedDict4Train = {}

          for featureType in featureTypes:
            if featureType == "DrugFingerPrint":
              batchX4DrugFingerPrint = self.insDataPro.data4DrugFingerPrint[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4DrugFingerPrint] = batchX4DrugFingerPrint

            elif featureType == "DrugPhy":
              batchX4DrugPhy = self.insDataPro.data4DrugPhy[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4DrugPhy] = batchX4DrugPhy

            elif featureType == "L1000":
              # A375
              batchX4L1000A375 = self.insDataPro.data4L1000A375[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4L1000A375] = batchX4L1000A375

              # HA1E
              batchX4L1000HA1E = self.insDataPro.data4L1000HA1E[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4L1000HA1E] = batchX4L1000HA1E

              # HT29
              batchX4L1000HT29 = self.insDataPro.data4L1000HT29[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4L1000HT29] = batchX4L1000HT29

              # MCF7
              batchX4L1000MCF7 = self.insDataPro.data4L1000MCF7[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4L1000MCF7] = batchX4L1000MCF7

              # PC3
              batchX4L1000PC3 = self.insDataPro.data4L1000PC3[self.xTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
              feedDict4Train[self.insCNNModel.xData4L1000PC3] = batchX4L1000PC3

          #batchY4Discriminator = self.insDataPro.label4Discriminator[self.yTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
          #feedDict4Train[self.insCNNModel.yLabel4Discriminator] = batchY4Discriminator

          batchY4Classification = self.insDataPro.label4Classification[self.yTrainIndex[ind4xyTrainIndex[j: j + batchSize]]]
          feedDict4Train[self.insCNNModel.yLabel4Classification] = batchY4Classification

          #ofsn, ofsl, ofsl2 = sess.run([self.insCNNModel.output4FixedSize, self.insCNNModel.output4FixedSize4LSTM, self.insCNNModel.output4FixedSize4LSTM2], feed_dict = feedDict4Train)
          #sess.run(self.insBiLSTM.trainStep, feed_dict = feedDict4Train)
          loss, _ = sess.run([self.insBiLSTM.loss4Classification, self.insBiLSTM.trainStep], feed_dict = feedDict4Train)
          #print("loss:", loss)

        for featureType in featureTypes:
          if featureType == "DrugFingerPrint":
            testX4DrugFingerPrint = self.insDataPro.data4DrugFingerPrint[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4DrugFingerPrint] = testX4DrugFingerPrint

          elif featureType == "DrugPhy":
            testX4DrugPhy = self.insDataPro.data4DrugPhy[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4DrugPhy] = testX4DrugPhy

          elif featureType == "L1000":
            testX4L1000A375 = self.insDataPro.data4L1000A375[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4L1000A375] = testX4L1000A375

            testX4L1000HA1E = self.insDataPro.data4L1000HA1E[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4L1000HA1E] = testX4L1000HA1E

            testX4L1000HT29 = self.insDataPro.data4L1000HT29[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4L1000HT29] = testX4L1000HT29

            testX4L1000MCF7 = self.insDataPro.data4L1000MCF7[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4L1000MCF7] = testX4L1000MCF7

            testX4L1000PC3 = self.insDataPro.data4L1000PC3[self.xTestIndex]
            feedDict4Test[self.insCNNModel.xData4L1000PC3] = testX4L1000PC3

        testY4Classification = self.insDataPro.label4Classification[self.yTestIndex]
        feedDict4Test[self.insCNNModel.yLabel4Classification] = testY4Classification

        score = sess.run(self.insBiLSTM.outputH4LSTM, feed_dict = feedDict4Test)

        hammingLoss = EvaluationMetric.getHammingLoss(score, testY4Classification)
        oneError = EvaluationMetric.getOneError(score, testY4Classification)
        coverage = EvaluationMetric.getCoverage(score, testY4Classification)
        rankingLoss = EvaluationMetric.getRankingLoss(score, testY4Classification)
        jaccardIndex = EvaluationMetric.getJaccardIndex(score, testY4Classification)
        averagePrecision = EvaluationMetric.getAveragePrecision(score, testY4Classification)

        print "  hammingLoss:", hammingLoss
        print "  oneError:", oneError
        print "  coverage:", coverage
        print "  rankingLoss:", rankingLoss
        print "  jaccardIndex:", jaccardIndex
        print "  averagePrecision:", averagePrecision

  # Traing and validation for discriminator
  def trainDiscriminator(self, insDiscriminator):
     self.xTrain, self.xTest, self.yTrain, self.yTest = \
        self.insDataPro.splitData2TrainAndVal()



