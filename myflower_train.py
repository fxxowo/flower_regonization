"""
@filename:myflower_train.py
@author:FengXing
@time:2024-05-01
"""
import time

import torch
from torch import optim, nn
from tqdm import tqdm

from flower_dataset import dataloaders
from myflower_model import Model

# 得到并保存神经网络模型checkpoint.pth
# （模型，数据，损失函数，优化器
if __name__ == "__main__":
    since = time.time()
    # 保存最好的准确率
    if torch.cuda.is_available():
        # 如果有，返回 CUDA 设备
        device = torch.device('cuda')
    else:
        # 否则返回 CPU 设备
        device = torch.device('cpu')
    model = Model().to(device)

    num_epochs = 25

    # 优化器设置
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    # （传入优化器，迭代了多少后要变换学习率，学习率要*多少）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
    running_corrects = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 把数据都取个遍
        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).to(device)
            # print(outputs.shape)
            index = torch.argmax(outputs, 1)
            res = torch.zeros(8, 102).to(device)
            for i in range(len(labels)):
                res[i][labels[i]] = 1
            loss = criterion(outputs, res)
            # print(index.shape)
            # print(labels.shape)
            for i in range(len(index)):
                if index[i] == labels[i]:
                    running_corrects += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), "model.pth")
    model = torch.load("./model.pth")
    print("accurate rate:" + str(float(running_corrects) / float(len(dataloaders['train'].dataset)) * 100))
