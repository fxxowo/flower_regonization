import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms

filename = 'checkpoint.pth'

"""""""""""""""图像增强（数据集预处理处理）open"""""""""""""""
# 图像增强：将数据集中照片进行旋转、翻折、放大...得到更多的数据
# ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
data_transforms = {  # data_transforms中指定了所有图像预处理操作,只需要修改训练集和验证集的名字后复制粘贴
    'train':
        transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                            transforms.CenterCrop(224),  # 从中心开始裁剪
                            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                            ]),
    'valid':
        transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]),
}
"""""""""""""""图像增强（数据集预处理处理）end"""""""""""""""

"""""""""""""""处理照片数据函数open"""""""""""""""


# 注意tensor的数据需要转换成numpy的格式，而且还需要还原回标准化的结果
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    #去除维度为1，降维
    image = image.numpy().squeeze()
    # 还原回h，w，c
    image = image.transpose(1, 2, 0)
    # 被标准化过了，还原非标准化样子
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


"""""""""""""""处理照片数据函数end"""""""""""""""

"""""""""""""""检测照片预处理函数open"""""""""""""""


def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作，再裁剪
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img


"""""""""""""""检测照片预处理函数end"""""""""""""""

"""""""""""""""展示一张照片函数open"""""""""""""""


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


"""""""""""""""展示一张照片函数end"""""""""""""""
