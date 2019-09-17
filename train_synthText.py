"""
Author: brooklyn

train with synthText
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from net.craft import CRAFT
import sys
from utils.cal_loss import cal_synthText_loss
from dataset.synthDataset import SynthDataset
import argparse
from eval import eval_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='CRAFT Train Fine-Tuning')
parser.add_argument('--gt_path', default='/media/brooklyn/EEEEE142EEE10425/SynthText/gt.mat', type=str, help='SynthText gt.mat')
parser.add_argument('--synth_dir', default='/media/brooklyn/EEEEE142EEE10425/SynthText', type=str, help='SynthText image dir')
parser.add_argument('--label_size', default=96, type=int, help='target label size')
parser.add_argument('--batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--test_batch_size', default=16, type=int, help='test data batch size')
parser.add_argument('--test_interval', default=40, type=int, help='test interval')
parser.add_argument('--max_iter', default=50000, type=int, help='max iteration')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=500, type=int, help='training epochs')
parser.add_argument('--test_iter', default=10, type=int, help='test iteration')
args = parser.parse_args()


image_transform = transforms.Compose([
    transforms.Resize((args.label_size * 2, args.label_size * 2)),
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.Resize((args.label_size,args.label_size)),
    transforms.ToTensor()

])

def train(net, epochs, batch_size, test_batch_size, lr, test_interval, max_iter, model_save_path, save_weight=True):

    train_data = SynthDataset(image_transform=image_transform,
                              label_transform=label_transform,
                              file_path=args.gt_path,
                              image_dir=args.synth_dir)
    steps_per_epoch = 1000
    #选取SynthText部分数据作为训练集
    train_num = batch_size * steps_per_epoch
    train_data = torch.utils.data.Subset(train_data, range(train_num))

    #划分训练集、验证集
    train_num = len(train_data)
    test_iter = 10
    val_num = test_batch_size * test_iter
    train_data, val_data = torch.utils.data.random_split(train_data, [train_num - val_num, val_num])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=False)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        print('epoch = ', epoch)
        for i, (images, labels_region, labels_affinity, _) in enumerate(train_loader):
            iter = epoch * steps_per_epoch + i
            #更新学习率
            if iter != 0 and iter % 10000 == 0:
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
            #反向传播
            optimizer.zero_grad()  #梯度清零
            loss.backward()  #计算梯度
            optimizer.step() #更新权重

            #打印损失和学习率信息
            if i % 10 == 0:
                print('i = ', i,': loss = ', loss.item(), ' lr = ', lr)
            #计算验证损失
            if i != 0 and i % test_interval == 0:
                test_loss = eval_net(net, val_loader, criterion, device)
                print('test: i = ', i, 'test_loss = ', test_loss, 'lr = ', lr)

                if save_weight:
                    torch.save(net.state_dict(), model_save_path + 'epoch_' + str(epoch) + '_iter' + str(i) + '.pth')
            #保存最后训练模型
            if iter == max_iter:
                if save_weight:
                    torch.save(net.state_dict(), model_save_path + 'final.pth')


if __name__ == "__main__":

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs  # 遍历数据集次数
    lr = args.lr  # 学习率
    test_interval = args.test_interval #测试间隔
    max_iter = args.max_iter

    net = CRAFT(pretrained=True)  # craft模型
    net = net.to(device)
    model_save_prefix = 'checkpoints/craft_netparam_'
    try:
        train(net=net,
              batch_size=batch_size,
              test_batch_size=test_batch_size,
              lr=lr,
              test_interval=test_interval,
              max_iter=max_iter,
              epochs=epochs,
              model_save_path=model_save_prefix)

    except KeyboardInterrupt:

        torch.save(net.state_dict(), 'INTERRUPTED1.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
