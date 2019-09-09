"""
    计算联合loss, 按照1:3使用OHEM
"""
import numpy as np
import torch

def get_ohem_num(labels_region, labels_affinity, device):
    """

    :param labels_region: 训练标签region score
    :param labels_affinity: 训练标签affinity score
    :return: 各像素标签的数量
    """
    numPos_region = torch.sum(torch.gt(labels_region, 0.1)).to(device)
    numNeg_region = torch.sum(torch.le(labels_region, 0.1)).to(device)
    numPos_affinity = torch.sum(torch.gt(labels_affinity, 0.1)).to(device)
    numNeg_affinity = torch.sum(torch.le(labels_affinity, 0.1)).to(device)
    #pos-neg ratio is 1:3
    if numPos_region * 3 < numNeg_region:
        numNeg_region = numPos_region * 3
    if numPos_affinity * 3 < numNeg_affinity:
        numNeg_affinity = numPos_affinity * 3
    return numPos_region, numNeg_region, numPos_affinity, numNeg_affinity

def cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device):
    """
    计算synthText强数据集的loss
    :param criterion: 损失函数
    :param score_text: 网络输出的region score
    :param score_link: 网络输出的affinity score
    :param labels_region: 训练标签region score
    :param labels_affinity: 训练标签affinity score
    :return: loss
    """

    numPos_region, numNeg_region, numPos_affinity, numNeg_affinity = get_ohem_num(labels_region, labels_affinity,
                                                                                  device)
    #联合损失 ohem loss
    #取全部的postive pixels的loss
    loss1_fg = criterion(score_text[np.where(labels_region > 0.1)], labels_region[np.where(labels_region > 0.1)])
    loss1_fg = torch.sum(loss1_fg) / numPos_region.to(torch.float32)
    loss1_bg = criterion(score_text[np.where(labels_region <= 0.1)], labels_region[np.where(labels_region <= 0.1)])
    #selects the pixel with high loss in the negative pixels
    loss1_bg, _ = loss1_bg.sort(descending=True)
    loss1_bg = torch.sum(loss1_bg[:numNeg_region]) / numNeg_region.to(torch.float32)
    loss2_fg = criterion(score_link[np.where(labels_affinity > 0.1)], labels_affinity[np.where(labels_affinity > 0.1)])
    loss2_fg = torch.sum(loss2_fg) / numPos_affinity.to(torch.float32)
    loss2_bg = criterion(score_link[np.where(labels_affinity <= 0.1)], labels_affinity[np.where(labels_affinity <= 0.1)])
    loss2_bg, _ = loss2_bg.sort(descending=True)
    loss2_bg = torch.sum(loss2_bg[:numNeg_affinity]) / numNeg_affinity.to(torch.float32)
    #联合loss
    loss = loss1_fg + loss1_bg + loss2_fg + loss2_bg

    return loss

def cal_fakeData_loss(criterion, score_text, score_link, labels_region, labels_affinity, sc_map, device):
    """

    :param criterion:损失函数
    :param score_text: 网络输出的region score
    :param score_link: 网络输出的affinity score
    :param labels_region: 训练标签region score
    :param labels_affinity: 训练标签affinity score
    :param sc_map: confidence map
    :return: loss
    """
    numPos_region, numNeg_region, numPos_affinity, numNeg_affinity = get_ohem_num(labels_region, labels_affinity,
                                                                                  device)
        #计算loss
    loss1_fg = criterion(score_text[np.where(labels_region > 0.1)], labels_region[np.where(labels_region > 0.1)])
    #添加 pixel-wise confidence map
    loss1_fg = loss1_fg * sc_map[np.where(labels_region > 0.1)]
    loss1_fg = torch.sum(loss1_fg) / numPos_region
    loss1_bg = criterion(score_text[np.where(labels_region <= 0.1)], labels_region[np.where(labels_region <= 0.1)])
    loss1_bg = loss1_bg * sc_map[np.where((labels_region <= 0.1))]
    #selects the pixel with high loss in the negative pixels
    loss1_bg, _ = loss1_bg.sort(descending=True)
    loss1_bg = torch.sum(loss1_bg[:numNeg_region]) / numNeg_region
    print('loss1_fg:', loss1_fg)
    print('loss1_bg:', loss1_bg)
    loss2_fg = criterion(score_link[np.where(labels_affinity > 0.1)], labels_affinity[np.where(labels_affinity > 0.1)])
    loss2_fg = loss2_fg * sc_map[np.where(labels_affinity > 0.1)]
    loss2_fg = torch.sum(loss2_fg) / numPos_affinity
    loss2_bg = criterion(score_link[np.where(labels_affinity <= 0.1)], labels_affinity[np.where(labels_affinity <= 0.1)])
    loss2_bg = loss2_bg * sc_map[np.where(labels_affinity <= 0.1)]
    loss2_bg, _ = loss2_bg.sort(descending=True)
    loss2_bg = torch.sum(loss2_bg[:numNeg_affinity]) / numNeg_affinity
    #联合loss
    loss = loss1_fg + loss1_bg + loss2_fg + loss2_bg

    return loss
