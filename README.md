## 数据

### 数据集

* 1、SynthText: 字符级标签的强数据集
* 2、ICDAR2013: word级标签的弱数据集

注：弱数据集可以是自己标注的文本框的数据集，icdar2013标签数据示例如下:
```
158 128 411 181 "Footpath"
443 128 501 169 "To"
64 200 363 243 "Colchester"
394 199 487 239 "and"
72 271 382 312 "Greenstead"
```
每行依次为每一个word边框的左上角列坐标xleft，左上角行坐标ytop，右下角列坐标xright，右下角行坐标ydown和文字信息。

如需使用自己标注的word-level数据集进行fine-tuning，则需要根据实际情况重写标签数据读取的代码。



### 标签

CRAFT模型训练标签共有两个：

1. Region score: 字符级的高斯热图

* I. 对于字符级的强数据集，直接由字符框生成高斯热图
* II. 对于word级的弱数据集，参考[伪标签生成](#生成伪标签)

2. Affinity score: 字符间连接的高斯热图

   参考如下[Affinity Box的生成](#Affinity Box的生成)

   

### Affinity Box的生成

1、连接Character Box对角线，得到2对三角形，上三角形（T）和下三角形（B），左三角形（L）和右三角形（R）。

2、字符1的2对三角形与字符2的两对三角形进行组合，产生4种组合情况，每组4个三角形。

3、每组4个三角形构成一个候选的Affinity Box。

4、选出其中面积最大且为凸四边形的Affinity Box。

如图所示：

![avatar](img/CRAFT高斯热图.png)



### 生成伪标签

对于只有Word级而无Character级标签的数据集（如ICDAR2013、ICDAR2015），需要生成Character级的标签。如图所示：

![avatar](img/伪标签.jpg)

1. 使用当前训练的模型预测出图像的Region Score Map。
2. 使用Word级的Box坐标crop出局部的Region Score Map。
3. 使用分水岭算法分割Region Score Map，得到Character Box的坐标。
4. 将Character Box的坐标转换回原坐标



## 训练



### 训练步骤

在强数据标签（SynthText）上进行强监督训练，迭代50k次。

在其他数据集（如ic13等有word级边框标注的数据集）上进行fine-tuning，此时要强标签数据和若标签数据混合训练。

### 训练技巧

fine-tuning期间，弱标签数据和强标签数据按照 1:5 的比例进行训练，以保证字符级标签的准确性。

### 原作者说明
initial lr: 1e-4, 每10k次迭代乘以0.8
batch size: 8 images for 1 GPU
ohem(pos-neg ratio): 1:3

## 代码说明
### 一、训练
训练分为两步：

* **1.强监督训练**

在强数据集(SynthText)上进行，迭代50k次

`sh train_synthText.sh`
* `--gt_path`: synthtext gt.mat路径
* `--synth_dir`： synthtext路径
* `--label_size`：标签热力图尺寸
* `--batch_size`：训练数据batch size
* `--test_batch_size`：测试batch size
* `--max_iter`：最大迭代次数
* `--lr`：初始学习率
* `--epochs` ：训练epochs
* `--test_interval`：测试间隔
* `--test_iter`：测试迭代次数

* **2.fine-tuning**

弱标签数据和强标签数据按照1:5的比例进行训练

`sh train_synthText.sh`
* `--gt_path`: synthtext gt.mat路径
* `--synth_dir`： synthtext路径
* `--ic13_root`：ic13数据集根目录
* `--label_size`：标签热力图尺寸
* `--batch_size`：训练数据batch
* `--test_batch_size`：验证batch
* `--cuda`：gpu训练
* `--pretrained_model`：预训练模型
* `--lr=3e-5`：初始学习率
* `--epochs=20` ：训练epochs
* `--test_interval`：测试间隔

使用的预训练模型可用原作者提供的预训练模型，位于`model/craft_mlt_25k.pth`; 

也可以采用自己在第一步强监督训练后得到的模型。建议采用原作者训好的预训练模型进行迁移学习。

**注：** 在只有word-level标注信息的情况下，建议在作者提供的预训练模型的基础上进行fine-tuning，得到所需的模型。

### 二、测试
```
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```
Arguments
* `--trained_model`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images

### 三、代码目录
```
./
├── basenet
│   ├── __init__.py
│   └── vgg16_bn.py				#vgg16网络
├── converts
│   ├── icdar2013_convert.py 	#ic13数据转换
│   └── synthText_convert.py	#synthText数据转换
├── dataset
│   ├── icdar2013_dataset.py	#ic13数据读取
│   └── synthDataset.py			#synthText数据读取
├── eval.py						#计算验证集loss
├── img
├── model
│   └── craft_mlt_25k.pth		#训好的模型
├── net
│   └── craft.py				#craft网络
├── README.md
├── test.py						#测试代码
├── train_finetune.py			# finetune	
├── train_finetune.sh			
├── train_synthText.py			#强监督训练
├── train_synthText.sh
└── utils
    ├── box_util.py
    ├── cal_loss.py				#计算loss
    ├── craft_utils.py
    ├── fake_util.py
    ├── file_utils.py
    ├── gaussian.py				#生成高斯热图
    ├── imgproc.py
    └── img_util.py
    ```
    
