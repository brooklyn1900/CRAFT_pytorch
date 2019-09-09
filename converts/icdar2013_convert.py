# -*- coding: utf-8 -*-
"""
读取并转化icdar13数据
"""
import os
import numpy as np

# label应为高斯热力图
def load_icdar2013(images_path, labels_path ):
    image_names = os.listdir(images_path)
    label_names = os.listdir(labels_path)
    image_names.sort()
    label_names.sort()
    return image_names, label_names

def get_wordsList(label_path):

    fh = open(label_path, 'r')
    word_boxes = []
    words = []
    for line in fh:
        line = line.rstrip()
        line = line.split()
        box = np.array(line[:4], dtype=int)
        #转换格式
        box = np.float32(
            [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]],
             [box[0], box[3]]])
        word = line[-1]
        word = word[1:-1]  # 去除双引号
        word_boxes.append(box)
        words.append(word)
    return word_boxes, words

