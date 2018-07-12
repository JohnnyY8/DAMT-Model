# coding=utf-8

import os
import random
import numpy as np
import tensorflow as tf

from dataPro import *
from cnnModel import *
from discriminator import *
from biLSTM import *
from modelTrainer import *

flags = tf.app.flags

flags.DEFINE_string(
    "gpuId",
    "0",
    "Which gpu is assigned.")

flags.DEFINE_string(
    "path4AllFiles",
    "./files",
    "Path for all files.")

flags.DEFINE_string(
    "path4Data",
    "./files/data",
    "Path for all data.")

flags.DEFINE_string(
    "path4TrainedModel",
    "./files/trainedModel",
    "Path for saving trained model.")

flags.DEFINE_string(
    "path4FinalValues",
    "./files/finalValues",
    "Path for saving final values.")

flags.DEFINE_float(
    "testSize",
    0.1,
    "Rate for validation data.")

flags.DEFINE_float(
    "learningRate",
    0.0001,
    "Learning rate.")

flags.DEFINE_float(
    "threshold4Classification",
    0.5,
    "Threshold for classification.")

flags.DEFINE_integer(
    "embeddingDimension",
    10,
    "Dimension for embedding.")

flags.DEFINE_integer(
    "batchSize",
    128,
    "Batch size for training.")

flags.DEFINE_integer(
    "displayStep",
    100,
    "Steps for displaying training procedure.")

flags.DEFINE_integer(
    "num4Data",
    6052,
    "Number of all data.")

flags.DEFINE_integer(
    "num4Features4Instance",
    7,
    "Number of features for each instance.")

flags.DEFINE_integer(
    "num4Classes",
    14,
    "Number of classeses.")

flags.DEFINE_integer(
    "num4Epoches",
    2000,
    "Epoches for training.")

flags.DEFINE_integer(
    "num4Hidden4LSTM",
    128,
    "Number of hidden units in LSTM.")

FLAGS = flags.FLAGS

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuId

  # Load data
  insDataPro = DataPro(FLAGS)

  # Get cnn model
  insCNNModel = CNNModel(FLAGS, insDataPro)

  # Get discriminator
  insDiscriminator = Discriminator(FLAGS, insCNNModel)

  # Get biLSTM
  insBiLSTM = BiLSTM(FLAGS, insCNNModel)
