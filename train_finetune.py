"""
Author: brooklyn

train with weak datasets like ICDAR2013
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from net.craft import CRAFT
import sys
from eval import copyStateDict, eval_net_finetune
from utils.cal_loss import cal_fakeData_loss, cal_synthText_loss
from dataset.synthDataset import SynthDataset
from dataset.icdar2013_dataset import Icdar2013Dataset
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
parser = argparse.ArgumentParser(description='CRAFT Train Fine-Tuning')
parser.add_argument('--gt_path', default='/media/brooklyn/EEEEE142EEE10425/SynthText/gt.mat', type=str, help='SynthText gt.mat')
parser.add_argument('--synth_dir', default='/media/brooklyn/EEEEE142EEE10425/SynthText', type=str, help='SynthText image dir')
parser.add_argument('--ic13_root', default='/home/brooklyn/ICDAR/icdar2013', type=str, help='icdar2013 data dir')
parser.add_argument('--label_size', default=96, type=int, help='target label size')
parser.add_argument('--batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--test_batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--pretrained_model', default='model/craft_mlt_25k.pth', type=str, help='pretrained model path')
parser.add_argument('--lr', default=3e-5, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=20, type=int, help='training epochs')
parser.add_argument('--test_interval', default=40, type=int, help='test interval')
args = parser.parse_args()


image_transform = transforms.Compose([
    transforms.Resize((args.label_size*2,args.label_size*2)),
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.Resize((args.label_size,args.label_size)),
    transforms.ToTensor()
])

def train(net, epochs, batch_size, test_batch_size, lr, test_interval, test_model_path, model_save_prefix, save_weight=True):

    ic13_data = Icdar2013Dataset(cuda=args.cuda,
                                 image_transform=image_transform,
                                 label_transform=label_transform,
                                 model_path=test_model_path,
                                 images_dir=os.path.join(args.ic13_root, 'train_images'),
                                 labels_dir=os.path.join(args.ic13_root, 'train_labels'))

    steps_per_epoch = 100

    ic13_length = len(ic13_data)
    synth_data = SynthDataset(image_transform=image_transform,
                              label_transform=label_transform,
                              file_path=args.gt_path,
                              image_dir=args.synth_dir)
    #弱数据集与强数据集比例1：5
    synth_data = torch.utils.data.Subset(synth_data, range(5*ic13_length))

    #合并弱数据集和强数据集
    fine_tune_data = torch.utils.data.ConcatDataset([synth_data, ic13_data])
    train_data, val_data = torch.utils.data.random_split(fine_tune_data, [5*ic13_length, ic13_length])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=False)
    print('len train data:', len(train_data))
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr)

    for epoch in range(epochs):
        print('epoch = ', epoch)
        for i, (images, labels_region, labels_affinity, sc_map) in enumerate(train_loader):

            images = images.to(device)
            labels_region = labels_region.to(device)
            labels_affinity = labels_affinity.to(device)
            sc_map = sc_map.to(device)
            labels_region = torch.squeeze(labels_region, 1)
            labels_affinity = torch.squeeze(labels_affinity, 1)

            #前向传播
            y, _ = net(images)
            score_text = y[:, :, :, 0]
            score_link = y[:, :, :, 1]
            sc_map = torch.squeeze(sc_map, 1)
            #强弱数据集分别计算损失
            if sc_map.size() == labels_region.size():
                loss = cal_fakeData_loss(criterion, score_text, score_link, labels_region, labels_affinity, sc_map,
                                         device)
            else:
                loss = cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device)

            #back propagation
            optimizer.zero_grad()  #梯度清零
            loss.backward()  #计算梯度
            optimizer.step() #更新权重
            if i % 10 == 0:
                print('i = ', i,': loss = ', loss.item())

            if i != 0 and i % test_interval == 0:
                test_loss = eval_net_finetune(net, val_loader, criterion, device)
                print('test: i = ', i, 'test_loss = ', test_loss, 'lr = ', lr)
                if save_weight:
                    torch.save(net.state_dict(), model_save_prefix + 'epoch_' + str(epoch) + '_iter' + str(i) + '.pth')

if __name__ == "__main__":

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs  # 遍历数据集次数
    lr = args.lr  # 学习率
    test_interval = args.test_interval #测试间隔
    pretrained_model = args.pretrained_model #预训练模型
    net = CRAFT(pretrained=True)  # craft模型

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(pretrained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(pretrained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net = net.to(device)
    net.train()
    model_save_prefix = 'finetune/craft_finetune_'
    try:
        train(net=net,
              epochs=epochs,
              batch_size=batch_size,
              test_batch_size=test_batch_size,
              lr=lr,test_interval=test_interval,
              test_model_path=pretrained_model,
              model_save_prefix =  model_save_prefix)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
