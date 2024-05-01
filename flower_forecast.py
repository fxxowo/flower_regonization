"""
基于 pytorch 搭建神经网络分类模型识别花的种类，输入一张花的照片，输出显示最有可能的前八种花的名称和该种花的照片。
"""

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

# 数据集，标签
from flower_dataset import dataloaders, cat_to_name
# 处理照片数据函数，检测照片预处理函数，展示一张照片函数
from flower_function import im_convert, process_image, imshow

"""""""""""""""flower_model中在本程序需要用到的参数和函数本程序中重新写一遍open"""""""""""""""
"""""""""""""""这样就不用调用flower_model程序，就不用再次训练模型了open"""""""""""""""
"""相关参数open"""
feature_extract = True
model_name = 'resnet'
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""相关参数open"""

"""""""""""""""冻结神经网络权重函数open"""""""""""""""


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:  # 这里为true
        for param in model.parameters():
            param.requires_grad = False  # 把除了最后全连接层，前面所有层权重冻结不能修改


"""""""""""""""冻结神经网络权重函数end"""""""""""""""

"""""""""""""""修改全连接层函数open"""""""""""""""


# （模型名字、得到类别个数、模型权重、
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        # 加载模型（下载）
        model_ft = models.resnet152(pretrained=use_pretrained)
        # 有选择性的选需要冻住哪些层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 取出最后一层
        num_ftrs = model_ft.fc.in_features
        # 重新做全连接层（102这里需要修改，因为本任务分类类别是102）
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


"""""""""""""""修改全连接层函数end"""""""""""""""

"""""""""""""""flower_model中在本程序需要用到的参数和函数本程序中重新写一遍end"""""""""""""""
"""""""""""""""这样就不用调用flower_model程序，就不用再次训练模型了end"""""""""""""""

"""""""""""""""加载测试模型open"""""""""""""""
# 加载模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
# GPU模式
model_ft = model_ft.to(device)
# 保存文件的名字
filename = 'checkpoint.pth'
# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])


"""""""""""""""加载测试模型end"""""""""""""""

"""""""""""""""设置检测图像数据open"""""""""""""""
image_path = 'flower_test.jpg'
img1 = process_image(image_path)  # 预处理一下
imshow(img1)  # 展示函数

# 得到一个batch的测试数据（一次处理8张照片），在这里用模型进行检测
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.__next__()
print(images.shape)
print(labels)
model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

# 得到概率最大的那个
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
"""""""""""""""设置检测图像数据open"""""""""""""""

"""""""""""""""设置展示界面open"""""""""""""""
# 设置展示预测结果，这张照片最像的前八类
fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

# 2*4展示出来
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()
"""""""""""""""设置展示界面end"""""""""""""""
