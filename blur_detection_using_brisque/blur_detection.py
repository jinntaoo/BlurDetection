# -*- coding: utf-8 -*-
# @Author  : Abel King
# @File    : blur_detection.py
# Code is far away from bug with the God Animal protecting

import cv2
import utils


def get_blur_score(image, fix_size=True):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if fix_size:
        image = utils.fix_image_size(image, expected_pixels=2E6)
    score = cv2.quality.QualityBRISQUE_compute(image, model_file_path=, range_file_path=)
    return score
