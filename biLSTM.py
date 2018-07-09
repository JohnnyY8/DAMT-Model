#coding=utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class BiLSTM(CommonModelFunc):

  def __init__(self, FLAGS, insCNNModel, insDataPro):
    self.FLAGS = FLAGS
    self.insCNNModel = insCNNModel

  # Get a graph for bidirectional LSTM
  def getBiLSTM(self):
    # Network Parameters
    num4Input = self.FLAGS.num4Input  # Same as the shape of output from hROIPooling
    timeSteps = self.FLAGS.timeSteps  # Same as number of features for each instance
    num4Hidden4LSTM = self.FLAGS.num4Hidden4LSTM  # The dimensionality of hidden output
    num4Classes = self.FLAGS.num4Classes  # The number of ATC classes
