# coding=utf-8

import os
import copy
import numpy as np

def cal_evaluation(prob, batch_ys, threshold):
    prob_temp = copy.deepcopy(np.array(prob))
    batch_ys_temp = copy.deepcopy(batch_ys)
    # batch_ys_temp[batch_ys_temp == -1] = 0
    ranking_loss = 0
    for batch_i, batch_y in enumerate(batch_ys_temp):
        pos_is = np.where(batch_y == 1)[0]   # 正样本所在的下标
        neg_is = np.where(batch_y == 0)[0]   # 负样本所在的下标
        one_rank_loss = 0
        for pos_i in pos_is:
            for neg_i in neg_is:
                if prob_temp[batch_i, pos_i] < prob_temp[batch_i, neg_i]:
                    one_rank_loss += 1
        ranking_loss += one_rank_loss / float(len(pos_is) * len(neg_is))
    ranking_loss /= len(batch_ys_temp)
    prob_temp[prob_temp >= threshold] = 1
    prob_temp[prob_temp < threshold] = 0
    hamming_loss = np.mean(np.count_nonzero(prob_temp != batch_ys_temp, axis=1) / float(len(prob_temp[0])))
    one_error = 1 - np.mean(batch_ys_temp[list(range(len(batch_ys_temp))), np.argmax(prob, axis=1)])
    sort_arg = np.argsort(prob, axis=1)     # 对概率进行从小到大排序，得到排序之前的下标
    coverage = 0.0
    for row_i, row in enumerate(sort_arg):
        for col_i, col in enumerate(row):
            if batch_ys_temp[row_i, col_i] == 1:
                coverage = coverage + len(label_names) - col_i - 1
                break
    coverage = coverage / len(batch_ys_temp)

    rank = np.argsort(-prob, axis=1)
    total_pre = 0
    for row_i, row in enumerate(rank):
        each_pre, pre_count = 0, 0
        for col_i, col in enumerate(row):
            if batch_ys[row_i, col] == 1:
                pre_count += 1
                each_pre += pre_count / float(col_i + 1)
        total_pre += each_pre / float(np.sum(batch_ys[row_i]))
    avg_pre = total_pre / len(rank)
    return hamming_loss, one_error, coverage, ranking_loss, avg_pre
