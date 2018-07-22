# coding=utf-8

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

class DataPro:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.featureTypes = np.array(["DrugFingerPrint", "DrugPhy", "L1000"])
    self.cls = np.array(["A375", "HA1E", "HT29", "MCF7", "PC3"])
    self.num4FeatureTypes = self.featureTypes.shape[0]
    self.num4Features4Instance = 0
    self.num4Features4DrugFingerPrint = 0
    self.num4Features4DrugPhy = 0
    self.num4Features4L1000 = 0

  # ===== Get feature data =====
  def getData(self):
    path4Data, path4Label = self.FLAGS.path4Data, self.FLAGS.path4Label

    for featureType in self.featureTypes:
      if featureType == "DrugFingerPrint":
        self.data4DrugFingerPrint = np.load(os.path.join(path4Data, "DrugFingerPrint", "DrugFingerPrint_6052SAMPLE.npy"))
        self.num4Features4DrugFingerPrint = self.data4DrugFingerPrint.shape[1]
        self.num4Features4Instance += 1
      elif featureType == "DrugPhy":
        self.data4DrugPhy = np.load(os.path.join(path4Data, "DrugPhy", "DrugPhy_6052SAMPLE.npy"))
        self.num4Features4DrugPhy = self.data4DrugPhy.shape[1]
        self.num4Features4Instance += 1
      elif featureType == "L1000":
        self.data4L1000A375 = np.load(os.path.join(path4Data, "L1000", "L1000_A375_6052SAMPLE.npy"))
        self.data4L1000HA1E = np.load(os.path.join(path4Data, "L1000", "L1000_HA1E_6052SAMPLE.npy"))
        self.data4L1000HT29 = np.load(os.path.join(path4Data, "L1000", "L1000_HT29_6052SAMPLE.npy"))
        self.data4L1000MCF7 = np.load(os.path.join(path4Data, "L1000", "L1000_MCF7_6052SAMPLE.npy"))
        self.data4L1000PC3 = np.load(os.path.join(path4Data, "L1000", "L1000_PC3_6052SAMPLE.npy"))
        self.num4Features4L1000 = self.data4L1000A375.shape[1]
        self.num4Features4Instance += 5

  # ===== Get feature types as label for discriminator =====
  def getLabels4Discriminator(self):
    for ind, featureType in enumerate(self.featureTypes):
      if featureType == "DrugFingerPrint":
        tempLabel = np.zeros([self.FLAGS.batchSize, self.num4FeatureTypes])
        tempLabel[:, ind] = 1

        if ind == 0:
          self.label4Discriminator = tempLabel
        else:
          self.label4Discriminator = np.vstack(
              (self.label4Discriminator, tempLabel))

      elif featureType == "DrugPhy":
        tempLabel = np.zeros([self.FLAGS.batchSize, self.num4FeatureTypes])
        tempLabel[:, ind] = 1

        if ind == 0:
          self.label4Discriminator = tempLabel
        else:
          self.label4Discriminator = np.vstack(
              (self.label4Discriminator, tempLabel))

      elif featureType == "L1000":
        tempLabel = np.zeros([self.FLAGS.batchSize * 5, self.num4FeatureTypes])
        tempLabel[:, ind] = 1

        if ind == 0:
          self.label4Discriminator = tempLabel
        else:
          self.label4Discriminator = np.vstack(
              (self.label4Discriminator, tempLabel))

  # ===== Get label for classification =====
  def getLabels4Classification(self):
    path4LabelNPY = os.path.join(self.FLAGS.path4Label, "Label_6052SAMPLE.npy")

    self.label4Classification = np.load(path4LabelNPY)

  # ===== Split data into train and validation set =====
  def splitData2TrainAndVal(self):
    # Split index, because too many feature types
    index = np.array(range(0, self.FLAGS.num4Data))

    xTrainIndex, xTestIndex, yTrainIndex, yTestIndex = train_test_split(
        index,
        index,
        test_size = self.FLAGS.testSize,
        random_state = 24)

    return xTrainIndex, xTestIndex, yTrainIndex, yTestIndex

