# -*- coding: utf-8 -*-
# @Author  : Abel King
# @File    : find_theta.py
# Code is far away from bug with the God Animal protecting

import utils
from blur_detection_using_svc import blur_detection_svd
from blur_detection_using_laplacian_variance import detection
import argparse
import pathlib
import numpy as np


def get_best_theta(metrics, theta_name):
    ret = float('-inf')
    _score = float('-inf')
    for metric in metrics:
        _tmp = list(metric.values())[0][theta_name]
        if _tmp != float('nan') and _tmp != float('-inf') and _tmp != float('inf') and _tmp > _score:
            _score = _tmp
            ret = list(metric.keys())[0]
    return ret


def main(args):
    blur_paths = sorted([str(each.absolute()) for each in (pathlib.Path(args.input_image_folder) / 'blur').glob('*')])
    label_dict = dict.fromkeys(blur_paths, 1)
    clear_paths = sorted([str(each.absolute()) for each in (pathlib.Path(args.input_image_folder) / 'clear').glob('*')])
    label_dict.update(dict.fromkeys(clear_paths, 0))

    score_dict = detection.get_batch_blur_degree(blur_paths)
    score_dict.update(detection.get_batch_blur_degree(clear_paths))

    paths, scores, labels = list(), list(), list()
    for _path in blur_paths + clear_paths:
        paths.append(_path)
        scores.append(score_dict[_path])
        labels.append(label_dict[_path])
    paths, scores, labels = np.asarray(paths), np.asarray(scores), np.asarray(labels)
    results = {'samples': paths, 'scores': scores, 'labels': labels}

    metrics = utils.range_theta(results, theta=0, length=0, theta_low=min(scores), theta_high=max(scores), times=10)
    best_theta = get_best_theta(metrics, theta_name='f1_score')
    print('best_theta: {}'.format(best_theta))

    metrics2 = utils.range_theta(results, theta=best_theta, length=5, times=20)
    best_theta2 = get_best_theta(metrics2, theta_name='f1_score')
    print('best_theta: {}'.format(best_theta2))
    print('metrics1: {} \n metrics2: {}'.format(metrics, metrics2))


def parse():
    parser = argparse.ArgumentParser(description='get some args')
    parser.add_argument('-ip', '--image_folder_path', dest='input_image_folder', type=str, required=True,
                        help='image folder path')
    parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=100.0,
                        help='blurry threshold')
    parser.add_argument('-f', '--fix_size', dest='fix_size', action='store_true',
                        help='fix the input image\'s size')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='set logging level to debug')
    parser.add_argument('-d', '--display', dest='display', action='store_true',
                        help='display images')
    return parser.parse_args()


if __name__ == "__main__":
    main(args=parse())
