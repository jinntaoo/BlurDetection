# -*- coding: utf-8 -*-
# @Author  : Abel King
# @File    : blur_detection.py

import cv2
import numpy as np
import tqdm
import heartrate as ht

# ht.trace(browser=True)


def fix_image_size(image, expected_pixels=2E6):
    ratio = float(expected_pixels) / float(image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def get_blur_degree(image_file, shape=(1024, 960), sv_num=10, fix_size=True):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if fix_size:
        img = fix_image_size(img)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv / total_sv


def get_blur_map(image_file, shape=(1024, 960), win_size=10, sv_num=3):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    new_img = np.zeros((img.shape[0] + win_size * 2, img.shape[1] + win_size * 2))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if i < win_size:
                p = win_size - i
            elif i > img.shape[0] + win_size - 1:
                p = img.shape[0] * 2 - i
            else:
                p = i - win_size
            if j < win_size:
                q = win_size - j
            elif j > img.shape[1] + win_size - 1:
                q = img.shape[1] * 2 - j
            else:
                q = j - win_size
            # print p,q, i, j
            new_img[i, j] = img[p, q]

    # cv2.imwrite('test.jpg', new_img)
    # cv2.imwrite('testin.jpg', img)
    blur_map = np.zeros((img.shape[0], img.shape[1]))
    max_sv = 0
    min_sv = 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            block = new_img[i:i + win_size * 2, j:j + win_size * 2]
            u, s, v = np.linalg.svd(block)
            top_sv = np.sum(s[0:sv_num])
            total_sv = np.sum(s)
            sv_degree = top_sv / total_sv
            if max_sv < sv_degree:
                max_sv = sv_degree
            if min_sv > sv_degree:
                min_sv = sv_degree
            blur_map[i, j] = sv_degree
    # cv2.imwrite('blurmap.jpg', (1 - blur_map) * 255)

    blur_map = (blur_map - min_sv) / (max_sv - min_sv)
    # cv2.imwrite('blurmap_norm.jpg', (1-blur_map)*255)
    return blur_map


def get_batch_blur_degree(image_files, shape, sv_num):
    scores = dict.fromkeys(image_files, None)
    for _path in tqdm.tqdm(image_files):
        scores[_path] = get_blur_degree(_path, shape, sv_num, fix_size=True)
    return scores


if __name__ == "__main__":
    import pathlib

    files = [str(tmp) for tmp in pathlib.Path('test/image/').glob('*/*')]
    for file in files:
        print(file, get_blur_degree(file, fix_size=True))
        out_file = file.replace('blur', 'blur_result')
        blur_map = get_blur_map(file)
        cv2.imwrite(out_file, (1 - blur_map) * 255)
