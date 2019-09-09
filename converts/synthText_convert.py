from utils.box_util import cal_affinity_boxes
import scipy.io as sio
from itertools import chain


def load_synthText(file_path):
    """
    加载synthText数据集
    :param file_path: synthText.mat路径
    :return: charBB字符边框， txt识别文字
    """
    file_mat = sio.loadmat(file_path)
    imnames = file_mat['imnames'][0,:]
    charBB = file_mat['charBB'][0,:]
    #wordBB = file_mat['wordBB'][0,:]
    txt = file_mat['txt'][0,:]
    return imnames, charBB, txt

def get_wordsList(words_lines):
    """
    分割字符串,去除换行和空格
    :param words_lines: SynthText数据集中的单张图片的txt信息
    :return: wordsList
    """
    wordsList = list()
    for words in words_lines:
        words = words.splitlines()
        for word in words:
            wordsList.append(word.split())
    wordsList = list(chain.from_iterable(wordsList))
    return wordsList


#输入一张图片的字符边框列表，字符串列表，输出affinity边框列表
def get_affinity_boxes_list(char_boxes_array, wordsList):
    """

    :param char_boxes_array: 字符边框矩阵
    :param wordsList: 从SynthText/gt.mat中读取到的文字列表
    :return: 字符边框列表和字间边框列表
    """
    # 字符索引，确定word中字符个数
    affinity_boxes_list = list()
    char_boxes_list = list()
    start = 0
    for word in wordsList:
        index = len(word)
        char_boxes = char_boxes_array[start:start + index, :, :]
        affinity_boxes_list.append(cal_affinity_boxes(char_boxes))
        char_boxes_list.append(char_boxes)
        start = start + index
    affinity_boxes_list = list(chain.from_iterable(affinity_boxes_list))
    char_boxes_list = list(chain.from_iterable(char_boxes_list))
    return char_boxes_list, affinity_boxes_list

