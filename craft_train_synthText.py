import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from net.craft import CRAFT
from collections import OrderedDict
import sys
from utils.cal_loss import cal_synthText_loss
from dataset import SynthDataset
import argparse
from eval import eval_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

parser = argparse.ArgumentParser(description='CRAFT Train Fine-Tuning')
parser.add_argument('--gt_path', default='/media/brooklyn/EEEEE142EEE10425/SynthText/gt.mat', type=str, help='SynthText gt.mat')
parser.add_argument('--synth_dir', default='/media/brooklyn/EEEEE142EEE10425/SynthText', type=str, help='SynthText image dir')
parser.add_argument('--label_size', default=96, type=int, help='target label size')
args = parser.parse_args()


image_transform = transforms.Compose([
    transforms.Resize((args.label_size * 2, args.label_size * 2)),
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.Resize((args.label_size,args.label_size)),
    transforms.ToTensor()

])

def train(net, epochs, batch_size, lr, test_iter, model_save_path, save_weight=True):

    train_data = SynthDataset(image_transform=image_transform, label_transform=label_transform, file_path=args.gt_path, image_dir=args.synth_dir)
    train_data = torch.utils.data.Subset(train_data, range(2000))

    #划分训练集、验证集
    train_num = len(train_data)
    val_num = int(train_num / 100)
    train_data, val_data = torch.utils.data.random_split(train_data, [train_num - val_num, val_num])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):

        for i, (images, labels_region, labels_affinity, _) in enumerate(train_loader):

            #更新学习率
            if i != 0 and i % 10000 == 0:
                for param in optimizer.param_groups:
                    param['lr'] *= 0.8

            images = images.to(device)
            labels_region = labels_region.to(device)
            labels_affinity = labels_affinity.to(device)

            labels_region = torch.squeeze(labels_region, 1)
            labels_affinity = torch.squeeze(labels_affinity, 1)
            #前向传播
            y, _ = net(images)
            score_text = y[:, :, :, 0]
            score_link = y[:, :, :, 1]
            #联合损失 ohem loss
            loss = cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device)
            #print('loss:', loss)
            #反向传播
            optimizer.zero_grad()  #梯度清零
            loss.backward()  #计算梯度
            optimizer.step() #更新权重

            #打印损失和学习率信息
            if i % 10 == 0:
                print('i = ', i,': loss = ', loss.item(), ' lr = ', lr)
            #计算验证损失
            if i != 0 and i % test_iter == 0:
                test_loss = eval_net(net, val_loader, criterion, device)
                print('test: i = ', i, 'test_loss = ', test_loss, 'lr = ', lr)

                if save_weight:
                    torch.save(net.state_dict(), model_save_path + 'iter' + str(i) + '.pth')


if __name__ == "__main__":

    batch_size = 2
    epochs = 1  # 遍历数据集次数
    lr = 0.0001  # 学习率
    test_iter = 40 #测试间隔
    pretrained_model = 'model/craft_mlt_25k.pth'
    net = CRAFT(pretrained=True)  # craft模型
    #net.load_state_dict(
    #    copyStateDict(torch.load(pretrained_model, map_location='cpu')))
    net = net.to(device)
    model_save_prefix = 'checkpoints/craft_netparam_'
    try:
        train(net=net, batch_size=batch_size, lr=lr, test_iter=test_iter, epochs=epochs, model_save_path=model_save_prefix)
    except KeyboardInterrupt:

        torch.save(net.state_dict(), 'INTERRUPTED1.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
