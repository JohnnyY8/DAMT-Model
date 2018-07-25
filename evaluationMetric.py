# coding=utf-8

import numpy as np

class EvaluationMetric:

  @staticmethod
  def getJaccardIndex(score, label, threshold = 0.5):
    '''
    预测为正的标签和真实标签之间交集除以他们之间的并集
    :param score: 预测概率矩阵
    :param label: 真实标签矩阵
    :param threshold: 阈值，默认0.5
    :return: 计算出来的Jaccard index
    '''
    tempScore = score.copy()
    tempScore[tempScore < threshold] = 0.0
    tempScore[tempScore >= threshold] = 1.0

    intersection = tempScore * label  # 计算预测结果和真实标签的交集
    union = tempScore + label  # 计算预测结果和真实标签的并集

    union[union > 1] = 1  # 将交集的元素置为1
    jaccardIndex = np.mean(intersection.sum(axis = 1) / union.sum(axis = 1))

    return jaccardIndex

  @staticmethod
  def getAveragePrecision(score, label):
    '''
    按照概率值从大到小排序，计算每个样本中每个真实标签之前的真实标签概率个数，最后求平均
    :param score: 预测概率矩阵
    :param label: 真实标签矩阵
    :return: 计算出来的Average Precision
    '''
    sortArg = np.argsort(-score, axis = 1)  # 概率值从大到小排序，得到下标
    count4AllPre = 0.0  # 最终结果的叠加值

    for ind4Row, row in enumerate(sortArg):
      count4EachData, count4PreEachLabel = 0.0, 0.0
      for ind4Col, col in enumerate(row):
        if label[ind4Row, col] == 1:
          count4PreEachLabel += 1
          count4EachData += count4PreEachLabel / ind4Col + 1
      count4AllPre += count4EachData / np.sum(label[ind4Row])
    averagePrecision = count4AllPre / sortArg.shape[0]

    return averagePrecision

  @staticmethod
  def getCoverage(score, label):
    '''
    按照概率值从大到小排序，计算概率值排序最靠后的真实标签的排序平均值
    :param score: 预测概率矩阵
    :param label: 真实标签矩阵
    :return: 计算出来的Coverage
    '''
    # 对概率进行从小到大排序，得到下标，这里从小到大是为了倒序查找提高效率
    sortArg = np.argsort(score, axis = 1)
    coverage = 0.0

    for ind4Row, row in enumerate(sortArg):
      for ind4Col, col in enumerate(row):
        if label[ind4Row, col] == 1:
          coverage += score.shape[1] - ind4Col - 1
          break
    coverage = coverage / score.shape[0]

    return coverage

  @staticmethod
  def getOneError(score, label):
    '''
    预测的概率值最大的标签不在真实标签集中的数量
    :param score: 预测概率矩阵
    :param label: 真实标签矩阵
    :return: 计算出来的One-Error
    '''
    oneError = 1 - np.mean(
        label[range(len(score)),
            np.argmax(score, axis = 1)])

    return oneError

  @staticmethod
  def getHammingLoss(score, label, threshold = 0.5):
    '''
    计算预测错误的标签占标签总数的比例
    :param score: 预测概率矩阵
    :param label: 真实标签矩阵
    :param threshold: 阈值，默认0.5
    :return: 计算出来的Hamming Loss
    '''
    tempScore = score.copy()
    tempLabel = label.copy()

    tempScore[tempScore < threshold] = 0
    tempScore[tempScore >= threshold] = 1

    hammingLoss = np.mean(
        np.count_nonzero(tempScore != tempLabel, axis = 1) / \
        float(tempScore.shape[1]))

    return hammingLoss

  @staticmethod
  def getRankingLoss(score, label):
    '''
    相关标签集合与不相关标签集合进行两两比较，
        然后统计相关标签的预测可能性比不相关标签额预测可能性要小的次数
    :param score: 预测概率矩阵
    :param label: 真实标签矩阵
    :return: 计算出来的Ranking Loss
    '''
    tempScore = score.copy()
    tempLabel = label.copy()
    rankingLoss = 0

    for ind4Row, row in enumerate(tempLabel):
      ind4Positives = np.where(row == 1)[0]  # 正样本所在的下标
      ind4Negatives = np.where(row == 0)[0]  # 负样本所在的下标
      count4EachData = 0

      for ind4Positive in ind4Positives:
        for ind4Negative in ind4Negatives:
          if tempScore[ind4Row, ind4Positive] < tempScore[ind4Row, ind4Negative]:
            count4EachData += 1

      rankingLoss += count4EachData / \
          float(ind4Positives.shape[0] * ind4Negatives.shape[0])

    rankingLoss /= tempLabel.shape[0]

    return rankingLoss
