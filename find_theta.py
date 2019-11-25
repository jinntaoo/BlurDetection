# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 下午3:08
# @Author  : Abel King
# @File    : find_theta.py
# @Software: PyCharm

import cv2
import numpy as np


class Metrics:
    """Two classes"""

    def __init__(self, results):
        """All members maybe np.array and shape is [n,] """
        self.results = results
        self.samples = results.samples
        self.preds = results.preds  # [[0, 0, 1]]
        self.labels = results.labels  # [[0, 1, 1]]
        self.num_total = len(results)
        self._get_init()

    def _get_init(self):
        xor = self.preds & self.labels
        n_true = self.num_total - sum(self.preds ^ self.labels)
        n_true_positive = sum(xor)
        n_true_negative = n_true - n_true_positive

        _index = np.argwhere(self.preds == 1)
        n_false_positive = sum([xor[ind1, ind2] for ind1, ind2 in _index])
        n_false_negative = self.num_total - n_true - n_false_positive

        self.true_positive_rate = n_true_positive / (n_true_positive+n_false_negative)  # sensitivity
        self.true_negative_rate = n_true_negative / (n_true_negative+n_false_positive)  # specificity
        self.false_positive_rate = n_false_positive / (n_false_positive+n_true_negative)  # 误检率
        self.false_negative_rate = n_false_negative / (n_false_negative+n_true_positive)
        self.precision = n_true_positive/(n_true_positive+n_false_positive)  # 查准
        self.recall = n_true_positive / (n_true_positive+n_false_negative)  # 查全
        self.accuracy = n_true / self.num_total


