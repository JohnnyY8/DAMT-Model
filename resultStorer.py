#coding=utf-8
import os
import numpy as np

class ResultStorer:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.loss = np.array([])
    self.trainAccu = np.array([])
    self.valAccu = np.array([])

  # Add loss
  def addLoss(self, loss):
    self.loss = np.append(self.loss, loss)

  # Save loss
  def saveLoss(self):
    np.save(os.path.join(self.FLAGS.accuSavePath, "loss.npy"), self.loss)

  # Add train accuracy
  def addTrainAccu(self, trainAccu):
    self.trainAccu = np.append(self.trainAccu, trainAccu)

  # Save train accuracy
  def saveTrainAccu(self):
    np.save(os.path.join(self.FLAGS.accuSavePath, "trainAccu.npy"), self.trainAccu)

  # Add validation accuracy
  def addValAccu(self, valAccu):
    self.valAccu = np.append(self.valAccu, valAccu)

  # Save validation accuracy
  def saveValAccu(self):
    np.save(os.path.join(self.FLAGS.accuSavePath, "valAccu.npy"), self.valAccu)

  # Save current train set
  def saveTrainSet(self, trainSet):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "trainSet.npy"), trainSet)

  # Save current train set index
  def saveTrainSetIndex(self, trainSetIndex):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "trainSetIndex.npy"), trainSet)

  # Save current train label
  def saveTrainLabel(self, trainLabel):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "trainLabel.npy"), trainLabel)

  # Save current train label index
  def saveTrainLabelIndex(self, trainLabelIndex):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "trainLabelIndex.npy"), trainLabel)

  # Save current validation set
  def saveValidationSet(self, validationSet):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "validationSet.npy"), validationSet)

  # Save current validation set index
  def saveValidationSetIndex(self, validationSetIndex):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "validationSetIndex.npy"), validationSet)

  # Save current validation label
  def saveValidationLabel(self, validationLabel):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "validationLabel.npy"), validationLabel)

  # Save current validation label index
  def saveValidationLabelIndex(self, validationLabel):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, "validationLabelIndex.npy"), validationLabel)

  # Save preActivation and activation value of validation set
  def saveValue(self, output, fileName):
    np.save(os.path.join(self.FLAGS.path4SaveFinalValue, fileName), output)

