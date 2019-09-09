import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from net.craft import CRAFT
import sys
from collections import OrderedDict
from utils.cal_loss import cal_fakeData_loss, cal_synthText_loss
from dataset import SynthDataset
from icdar2013_dataset import Icdar2013Dataset
import argparse

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
parser.add_argument('--ic13_root', default='/home/brooklyn/ICDAR/icdar2013', type=str, help='icdar2013 data dir')
args = parser.parse_args()


image_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.Resize((192,192)),
    transforms.ToTensor()
])

def train(net, epochs, batch_size, lr, test_model_path, model_save_path, save_weight=True):

    ic13_data = Icdar2013Dataset(image_transform=image_transform, label_transform=label_transform, model_path=test_model_path,
                                  images_dir=os.path.join(args.ic13_root, 'train_images'),
                                  labels_dir=os.path.join(args.ic13_root, 'train_labels'))
    ic13_length = len(ic13_data)
    synth_data = SynthDataset(image_transform=image_transform, label_transform=label_transform,file_path=args.gt_path,image_dir=args.synth_dir)
    #弱数据集与强数据集比例1：5
    synth_data = torch.utils.data.Subset(synth_data, range(5*ic13_length))

    #合并弱数据集和强数据集
    fine_tune_data = torch.utils.data.ConcatDataset([synth_data, ic13_data])
    fine_tune_loader = torch.utils.data.DataLoader(fine_tune_data, batch_size, shuffle=True)
    print('len fine_data:', len(fine_tune_data))
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr)

    iter = 0 #迭代次数
    for epoch in range(epochs):
        for i, (images, labels_region, labels_affinity, sc_map) in enumerate(fine_tune_loader):

            images = images.to(device)
            labels_region = labels_region.to(device)
            labels_affinity = labels_affinity.to(device)

            labels_region = torch.squeeze(labels_region, 1)
            labels_affinity = torch.squeeze(labels_affinity, 1)

            #前向传播
            y, _ = net(images)
            score_text = y[:, :, :, 0]#.cpu().data.numpy()
            score_link = y[:, :, :, 1]#.cpu().data.numpy()

            sc_map = torch.squeeze(sc_map, 1)
            #强弱数据集分别计算损失
            if sc_map.size() == labels_region.size():
                loss = cal_fakeData_loss(criterion, score_text, score_link, labels_region, labels_affinity, sc_map,
                                         device)
            else:
                loss = cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device)

            #print('loss:', loss.item())
            #反向传播
            optimizer.zero_grad()  #梯度清零
            loss.backward()  #计算梯度
            optimizer.step() #更新权重
            if i % 10 == 0:
                print('i = ', i,': loss = ', loss.item())
    if save_weight:
        torch.save(net.state_dict(), model_save_path)

#数据集下好之后要加载数据

if __name__ == "__main__":

    batch_size = 2
    epochs = 1  # 遍历数据集次数
    lr = 0.00003  # 学习率
    pretrained_model = 'model/craft_mlt_25k.pth'
    net = CRAFT(pretrained=True)  # craft模型
    net.load_state_dict(copyStateDict(torch.load(pretrained_model, map_location='cpu')))
    net = net.to(device)
    net.train()
    model_save_path = 'net_param.pth'
    try:
        train(net=net,batch_size=batch_size,lr=lr,epochs=epochs, test_model_path=pretrained_model, model_save_path = model_save_path)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
