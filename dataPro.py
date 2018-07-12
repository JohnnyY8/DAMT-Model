# coding=utf-8

import os
import numpy as np
import tensorflow as tf

class DataPro:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.featureTypes = np.array(["DrugFingerPrint", "DrugPhy", "L1000"])
    self.cls = np.array(["A375", "HA1E", "HT29", "MCF7", "PC3"])
    self.data, self.label4Discriminator, self.label4Classification = \
        np.array([]), np.array([]), np.array([])

  # ===== Get several feature type data =====
  def getFeatureData(self):
    path4Sample = os.path.join(self.path4Data, "Sample")

    for featureType in self.featureTypes:
      if featureType == "L1000":
        for cl in cls:
          path4DataNPY = os.path.join(path4Sample, "L1000", "L1000_" + cl + "_6052SAMPLE.npy")
          self.stackData(self.data, np.load(path4DataNPY)
      else:
        path4DataNPY = os.path.join(path4Sample, featureType + "_6052SAMPLE.npy")
        self.stackData(self.data, np.load(path4DataNPY))
    self.num4Data = np.load(path4DataNPY).shape[0]

  # ===== Get feature types as label for discriminator =====
  def getFeatureTypes(self):
    self.num4FeatureTypes = self.featureTypes.shape[0]

    for ind, featureType in enumerate(self.featureTypes):
      if featureType == "L1000":
        for cl in cls:
          tempLabel = np.zeros([self.num4Data, self.num4FeatureTypes])
          tempLabel[:, ind] = 1
          self.stackData(self.label4FeatureTypes, tempLabel)
      else:
        tempLabel = np.zeros([self.num4Data, self.num4FeatureTypes])
        tempLabel[:, ind] = 1
        self.stackData(self.label4FeatureTypes, tepmLabel)

  # ===== Stack all feature type data or label =====
  # The interval is 6052 in this study
  def stackData(self, old, new):
    if old.shape[0] == 0:
      old = new
    else:
      old = np.vstack((old, new))

