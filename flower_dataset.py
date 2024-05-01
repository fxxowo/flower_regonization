import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
# pip install torchvision 需要提前安装好这个模块
from torchvision import transforms, models, datasets
# https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# 图像增强（数据集预处理处理）
from flower_function import data_transforms

"""""""""""""""读取训练集、测试集open"""""""""""""""

data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

"""""""""""""""读取训练集、测试集end"""""""""""""""

"""""""""""""""构建神经网络的数据集open"""""""""""""""
"""都存到dataloaders中"""

# batch_size是设置一次训练多少张照片
batch_size = 8  # 设置越大需要的显存越大
# (os.path.join(data_dir, x), data_transforms[x])，（两个文件夹传进去，传入数据增强的数据）
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# class_name 是训练集
class_names = image_datasets['train'].classes

"""
print(image_datasets)
print(dataloaders)
print(dataset_sizes)
"""

# 数据集中类别按照123456...标号，文件是各标号对应的名称
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
"""
print(cat_to_name) #打印标号集
"""

"""""""""""""""构建神经网络的数据集end"""""""""""""""

"""""""""""""""打印照片操作open"""""""""""""""
"""
fig=plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(dataloaders['valid'])
inputs, classes = dataiter.next()

#做图，print出来
for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    plt.imshow(im_convert(inputs[idx]))
plt.show()
"""
"""""""""""""""打印照片操作end"""""""""""""""
