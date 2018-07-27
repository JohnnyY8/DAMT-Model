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
    "./files/data/Sample",
    "Path for all data.")

flags.DEFINE_string(
    "path4Label",
    "./files/data/Label",
    "Path for all label.")

flags.DEFINE_string(
    "path4TrainedModel",
    "./files/trainedModel",
    "Path for saving trained model.")

flags.DEFINE_string(
    "path4SaveFinalValues",
    "./files/finalValues",
    "Path for saving final values.")

flags.DEFINE_float(
    "testSize",
    0.1,
    "Rate for validation data.")

flags.DEFINE_float(
    "learningRate",
    0.001,
    "Learning rate.")

flags.DEFINE_float(
    "threshold4Classification",
    0.5,
    "Threshold for classification.")

flags.DEFINE_integer(
    "num4InputChannels4Input",
    1,
    "Number of input channels for input layer.")

flags.DEFINE_integer(
    "num4InputChannels4Conv1",
    1,
    "Number of input channels for conv1 layer.")

flags.DEFINE_integer(
    "num4OutputChannels4Conv1",
    8,
    "Number of output channels for conv1 layer.")

flags.DEFINE_integer(
    "num4InputChannels4Conv2",
    8,
    "Number of input channels for conv2 layer.")

flags.DEFINE_integer(
    "num4OutputChannels4Conv2",
    12,
    "Number of output channels for conv2 layer.")

flags.DEFINE_integer(
    "num4InputChannels4Conv3",
    12,
    "Number of input channels for conv3 layer.")

flags.DEFINE_integer(
    "num4OutputChannels4Conv3",
    24,
    "Number of output channels for conv3 layer.")

flags.DEFINE_integer(
    "embeddingDimension",
    10,
    "Dimension for embedding.")

flags.DEFINE_integer(
    "batchSize",
    100,
    "Batch size for training.")

flags.DEFINE_integer(
    "batchSize4Discriminator",
    5,
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
    "num4Classes",
    14,
    "Number of classeses.")

flags.DEFINE_integer(
    "num4Epoches",
    200,
    "Epoches for training.")

flags.DEFINE_integer(
    "num4Epoches4Discriminator",
    10,
    "Epoches for training.")

flags.DEFINE_integer(
    "num4HiddenUnits4LSTM",
    128,
    "Number of hidden units in LSTM.")

FLAGS = flags.FLAGS

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuId

  # Load data
  insDataPro = DataPro(FLAGS)
  insDataPro.getData()

  # Get cnn model
  insCNNModel = CNNModel(FLAGS, insDataPro)
  insCNNModel.getCNNModel()

  # Get model trainer instance
  insModelTrainer = ModelTrainer(FLAGS, insDataPro, insCNNModel)

  # Get discriminator
  insDiscriminator = Discriminator(FLAGS, insDataPro, insCNNModel)
  insDiscriminator.getDiscriminator()
  insModelTrainer.trainDiscriminator(insDiscriminator)

  # Get biLSTM
  insDataPro.getLabels4Classification()
  insBiLSTM = BiLSTM(FLAGS, insDataPro, insCNNModel)
  insBiLSTM.getBiLSTM()
  insModelTrainer.trainLSTM(insBiLSTM)
