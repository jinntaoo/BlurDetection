# -*- coding: utf-8 -*-
# @Author  : Abel King
# @File    : find_theta.py

import numpy as np
import tqdm
import cv2


def fix_image_size(image, expected_pixels=2E6):
    ratio = float(expected_pixels) / float(image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


class Metrics:
    """Two classes"""

    def __init__(self, results):
        """All members maybe np.array and shape is [n,] """
        self.results = results
        self.samples = results['samples']
        self.preds = results['preds']  # [[0, 0, 1]]
        self.labels = results['labels']  # [[0, 1, 1]]
        self.num_total = len(results['samples'])
        self.n_positive = sum(results['labels'])
        self.n_negative = self.num_total - self.n_positive
        self._get_init()

    def _get_init(self):
        xor = self.preds & self.labels
        xor = self.preds ^ self.labels
        n_true = self.num_total - sum(xor)
        n_true_positive = sum(self.preds & self.labels)
        n_true_negative = n_true - n_true_positive

        _index = np.argwhere(self.preds == 1).tolist()
        n_false_positive = sum([xor[ind[0]] for ind in _index])
        n_false_negative = self.num_total - n_true - n_false_positive
        self.true_positive_rate = n_true_positive / self.n_positive if self.n_positive != 0 else float('nan')  # sensitivity
        self.true_negative_rate = n_true_negative / self.n_negative if self.n_negative != 0 else float('nan')  # specificity
        self.false_positive_rate = n_false_positive / self.n_negative if self.n_negative != 0 else float('nan')  # 误检率
        self.false_negative_rate = n_false_negative / self.n_positive if self.n_positive != 0 else float('nan')
        self.precision = n_true_positive / (n_true_positive + n_false_positive) if n_true_positive+n_false_positive != 0 else float('nan')  # 查准
        self.recall = n_true_positive / (n_true_positive + n_false_negative) if n_true_positive+n_false_negative != 0 else float('nan')  # 查全
        self.accuracy = n_true / self.num_total
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall) \
            if type(self.accuracy) is not type(float('nan')) and type(self.recall) is not type(float('nan')) else float('nan')
        self.metric_dict = ({'tpr': self.true_positive_rate, 'tnr': self.true_negative_rate,
                             'fpr': self.false_positive_rate, 'fnr': self.false_negative_rate,
                             'precision': self.precision, 'recall': self.recall,
                             'accuracy': self.accuracy, 'f1_score': self.f1_score})


def get_preds(theta, scores, positive_type=0):
    if positive_type == 0:
        return np.where(scores >= theta, 1, 0)
    elif positive_type == 1:
        return np.where(scores >= theta, 0, 1)
    else:
        raise Exception('Sth. Wrong')


def range_theta(results, theta, length, theta_low=0, theta_high=0, times=10, positive_type=1):
    samples, scores, labels = results['samples'], results['scores'], results['labels']
    metric_results = list()
    if theta_low == theta_high == 0:
        theta_low, theta_high = theta - length, theta + length
    for theta in tqdm.tqdm(np.arange(theta_low, theta_high, (theta_high - theta_low) / times)):
        preds = get_preds(theta, scores, positive_type)
        metrics = Metrics({'samples': samples, 'preds': preds, 'labels': labels})
        metric_results.append({theta: metrics.metric_dict})
        # print('theta: {0}'.format(theta))
    return metric_results
