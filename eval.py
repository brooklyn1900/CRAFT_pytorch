"""
计算验证集的loss
"""
import torch
from utils.cal_loss import cal_synthText_loss

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