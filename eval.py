"""
计算验证集的loss
"""
from collections import OrderedDict
import torch
from utils.cal_loss import cal_synthText_loss,cal_fakeData_loss

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def eval_net(net, val_loader, criterion, device):

    net.eval()
    loss_total = 0
    for i, (images, labels_region, labels_affinity, _) in enumerate(val_loader):

        images = images.to(device)
        labels_region = labels_region.to(device)
        labels_affinity = labels_affinity.to(device)

        labels_region = torch.squeeze(labels_region, 1)
        labels_affinity = torch.squeeze(labels_affinity, 1)
        # 前向传播
        y, _ = net(images)
        score_text = y[:, :, :, 0]
        score_link = y[:, :, :, 1]
        # 联合损失 ohem loss
        loss = cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device)
        loss_total += loss.item()

    return loss_total / (i + 1)

def eval_net_finetune(net, val_loader, criterion, device):

    net.eval()
    loss_total = 0
    for i, (images, labels_region, labels_affinity, sc_map) in enumerate(val_loader):

        images = images.to(device)
        labels_region = labels_region.to(device)
        labels_affinity = labels_affinity.to(device)
        sc_map = sc_map.to(device)
        labels_region = torch.squeeze(labels_region, 1)
        labels_affinity = torch.squeeze(labels_affinity, 1)
        # 前向传播
        y, _ = net(images)
        score_text = y[:, :, :, 0]
        score_link = y[:, :, :, 1]
        sc_map = torch.squeeze(sc_map, 1)
        # 联合损失 ohem loss
        # 强弱数据集分别计算损失
        if sc_map.size() == labels_region.size():
            loss = cal_fakeData_loss(criterion, score_text, score_link, labels_region, labels_affinity, sc_map,
                                     device)
        else:
            loss = cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device)

        loss_total += loss.item()

    return loss_total / (i + 1)