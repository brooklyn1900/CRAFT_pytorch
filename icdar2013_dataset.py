import torch
from PIL import Image
import os
import numpy as np
from utils.gaussian import GaussianGenerator
import cv2
from net.craft import CRAFT
from converts.icdar2013_convert import load_icdar2013, get_wordsList
from collections import OrderedDict
from utils.fake_util import crop_image, divide_region, watershed, find_box, cal_confidence
from utils.box_util import reorder_points, cal_affinity_boxes
from utils.img_util import img_normalize, load_image
from itertools import chain
from utils import imgproc
from torch.autograd import Variable

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

#icdar2013 dataset类
class Icdar2013Dataset(torch.utils.data.Dataset):
    def __init__(self, image_transform=None, label_transform=None, target_transform=None, model_path=None, images_dir=None, labels_dir=None):
        super(Icdar2013Dataset, self).__init__() #继承父类构造方法
        self.model_path = model_path
        #图片名和标签数据（不是标签名）
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_names, self.label_names = load_icdar2013(self.images_dir, self.labels_dir)
        self.craft = CRAFT()
        self.craft.load_state_dict(copyStateDict(torch.load(self.model_path,map_location='cpu')))
        self.craft.eval()

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_names)

    # label应为高斯热力图
    def __getitem__(self, idx):
        fn = self.image_names[idx]
        image = load_image(os.path.join(self.images_dir, fn))
        label_name = self.label_names[idx]
        word_boxes, words = get_wordsList(os.path.join(self.labels_dir, label_name))
        char_boxes_list, affinity_boxes_list, confidence_list = self.get_affinity_boxes_list(image, word_boxes, words)
        height, width = image.shape[:2] #opencv方式
        heat_map_size = (height, width)
        #get pixel-wise confidence map
        sc_map = self.get_sc_map(heat_map_size, word_boxes, confidence_list) * 255
        region_scores = self.get_region_scores(heat_map_size, char_boxes_list) * 255
        affinity_scores = self.get_region_scores(heat_map_size, affinity_boxes_list) * 255

        #opencv转为PIL.Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

    def fake_char_boxes(self, src, word_box, word_length):
        img, src_points, crop_points = crop_image(src, word_box, dst_height=64.)
        h, w = img.shape[:2]
        if min(h, w) == 0:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
            return region_boxes, confidence
        img = img_normalize(img)
        region_score, affinity_score = self.test_net(self.craft, img, False)
        heat_map = region_score * 255.
        heat_map = heat_map.astype(np.uint8)
        marker_map = watershed(heat_map)
        region_boxes = find_box(marker_map)
        confidence = cal_confidence(region_boxes, word_length)
        if confidence <= 0.5:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
        else:
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]

        return region_boxes, confidence

    def get_affinity_boxes_list(self, image, word_boxes, words):
        char_boxes_list = list()
        affinity_boxes_list = list()
        confidence_list = list()

        for word_box, word in zip(word_boxes, words):
            char_boxes, confidence = self.fake_char_boxes(image, word_box, len(word))
            affinity_boxes = cal_affinity_boxes(char_boxes)
            affinity_boxes_list.append((affinity_boxes))
            char_boxes_list.append(char_boxes)
            confidence_list.append(confidence)

        char_boxes_list = list(chain.from_iterable(char_boxes_list))
        affinity_boxes_list = list(chain.from_iterable(affinity_boxes_list))
        return char_boxes_list, affinity_boxes_list, confidence_list

    def get_region_scores(self, heat_map_size, char_boxes_list):
        # 高斯热力图
        gaussian_generator = GaussianGenerator()
        char_boxes_list = np.array(char_boxes_list, dtype=np.float32)
        region_scores = gaussian_generator.gen(heat_map_size, char_boxes_list)
        return region_scores

    def get_sc_map(self, heat_map_size,word_boxes, confidence_list):

        """
        :param heat_map_size:
        :param word_boxes:
        :param confidence_list:
        :return: pixel-wise confidence map Sc
        """
        word_boxes = np.array(word_boxes, dtype=int)
        sc_map = np.ones(heat_map_size, dtype=np.float32)
        for (word_box, confidence) in zip(word_boxes, confidence_list):
            x_left = word_box[0, 0]
            y_top = word_box[0, 1]
            x_right = word_box[2,0]
            y_down = word_box[2,1]
            sc_map[y_top:y_down, x_left:x_right] = confidence
        return sc_map

    def test_net(self, net, image, cuda):

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=1.5)
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()
        # forward pass
        with torch.no_grad():
            y, _ = net(x)
        #    make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        return score_text, score_link