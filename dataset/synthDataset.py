import torch
from PIL import Image
import os
import numpy as np
from utils.gaussian import GaussianGenerator
from converts.synthText_convert import *

#重写dataset类
class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, image_transform=None, label_transform=None, target_transform=None, file_path=None, image_dir=None):
        super(SynthDataset, self).__init__() #继承父类构造方法

        #图片名和标签数据（不是标签名）
        # 加载syntnText数据集
        imnames, charBB, txt = load_synthText(file_path)
        self.imnames = imnames
        self.charBB = charBB
        self.txt = txt
        self.image_dir = image_dir #训练数据文件夹地址
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.target_transform = target_transform
        self.sc_map = torch.ones(1)
    def __len__(self):
        return len(self.imnames)

    # label应为高斯热力图
    def __getitem__(self, idx):
        imname = self.imnames[idx].item()
        image = Image.open(os.path.join(self.image_dir, imname))

        #numpy ndarray格式
        char_boxes_array = np.array(self.charBB[idx])
        char_boxes_array = char_boxes_array.swapaxes(0,2)
        #生成affinity边框列表
        word_lines = self.txt[idx]
        word_list = get_wordsList(word_lines)   #文字列表
        char_boxes_list, affinity_boxes_list = get_affinity_boxes_list(char_boxes_array, word_list)

        width, height = image.size
        heat_map_size = (height, width)
        region_scores = self.get_region_scores(heat_map_size, char_boxes_list) * 255
        affinity_scores = self.get_region_scores(heat_map_size, affinity_boxes_list) * 255
        sc_map = np.ones(heat_map_size, dtype=np.float32) * 255
        #numpy.ndarray转为PIL.Image
        region_scores = Image.fromarray(np.uint8(region_scores))
        affinity_scores = Image.fromarray(np.uint8(affinity_scores))
        sc_map = Image.fromarray(np.uint8(sc_map))
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.label_transform is not None:
            region_scores = self.label_transform(region_scores)
            affinity_scores = self.label_transform(affinity_scores)
            sc_map = self.label_transform(sc_map)
        return image, region_scores, affinity_scores, sc_map

    #获取图片的高斯热力图
    def get_region_scores(self, heat_map_size, char_boxes_list):
        # 高斯热力图
        gaussian_generator = GaussianGenerator()
        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list)
        return region_scores

