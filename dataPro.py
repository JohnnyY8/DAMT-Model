# coding=utf-8

import os
import numpy as np
import tensorflow as tf

class DataPro:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.featureTypes = np.array(["DrugFingerPrint", "DrugPhy", "L1000"])
    self.cls = np.array(["A375", "HA1E", "HT29", "MCF7", "PC3"])
    self.num4Features4Instance = 0

  # ===== Get feature data =====
  def getData(self):
    path4Sample = os.path.join(self.path4Data, "Sample")

    for featureType in self.featureTypes:
      if featureType == "DrugFingerPrint":
        self.data4DrugFingerPrint = np.load(os.path.join(path4Sample, "DrugFingerPrint", "DrugfingerPrint_6052SAMPLE.npy"))
        self.num4Features4Instance += 1
      elif featureType == "DrugPhy":
        self.data4DrugPht = np.load(os.path.join(path3Sample, "DrugPhy", "DrugPhy_6052SAMPLE.npy"))
        self.num4Features4Instance += 1
      elif featureType == "L1000":
        self.data4L10004A375 = np.load(os.path.join(path4Sample, "L1000", "L1000_A375_6052SAMPLE.npy"))
        self.data4L10004HA1E = np.load(os.path.join(path4Sample, "L1000", "L1000_HA1E_6052SAMPLE.npy"))
        self.data4L10004HT29 = np.load(os.path.join(path4Sample, "L1000", "L1000_HT29_6052SAMPLE.npy"))
        self.data4L10004MCF7 = np.load(os.path.join(path4Sample, "L1000", "L1000_MCF7_6052SAMPLE.npy"))
        self.data4L10004PC3 = np.load(os.path.join(path4Sample, "L1000", "L1000_PC3_6052SAMPLE.npy"))
        self.num4Features4Instance += 5

  # ===== Get feature types as label for discriminator =====
  def getLabels4Discriminator(self):
    self.num4FeatureTypes = self.featureTypes.shape[0]

    for ind, featureType in enumerate(self.featureTypes):
      tempLabel = np.zeros([self.FLAGS.num4Data, self.num4FeatureTypes])
      tempLabel[:, ind] = 1

      if featureType == "DrugFingerPrint":
        self.label4DrugFingerPrint4Discriminator = tempLabel
      elif featureType == "DrugPhy":
        self.label4DrugPhy4Discriminator = tempLabel
      elif featureType == "L1000":
        self.label4L10004A3754Discriminator = tempLabel
        self.label4L10004HA1E4Discriminator = tempLabel
        self.label4L10004HT294Discriminator = tempLabel
        self.label4L10004MCF74Discriminator = tempLabel
        self.label4L10004PC34Discriminator = tempLabel

  # ===== Get label for classification =====
  def getLabels4Classification(self):
    path4LabelNPY = os.path.join(self.path4Data, "Label", "Label_6052SAMPLE.npy")

    self.label4Classification = np.load(path4LabelNPY)

