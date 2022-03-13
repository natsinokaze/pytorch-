import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)

dataloader = DataLoader(dataset, batch_size=1)

class Seq(nn.Module):
    def __init__(self):
        super(Seq, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 3@32*32 --> 32@32*32
            MaxPool2d(2),  # 32@32*32 --> 32@16*16
            Conv2d(32, 32, 5, padding=2),  # 32@16*16 --> 32@16*16
            MaxPool2d(2),  # 32@16*16 --> 32@8*8
            Conv2d(32, 64, 5, padding=2),  # 32@8*8 --> 64@8*8
            MaxPool2d(2),  # 64@8*8 --> 64@4*4
            Flatten(),
            Linear(1024, 64),  # 第一项 一定要正确这里是1024的原因是经过maxpool3后有64*4*4=1024个元素
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

seq = Seq()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(seq.parameters(),lr=0.01)  # ir(学习速率)设置太高会有较大误差 设置过小会影响计算效率
for epoch in range(20):  # 表示循环所有数据20遍
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = seq(imgs)  # 得到的结果为batch_size=1的10个类的线性值 注意不是概率
        result_loss = loss(outputs, targets)
        optim.zero_grad()  # 每次循环前先清空梯度（grad）避免影响本次的梯度o

        result_loss.backward()
        optim.step()
        running_loss += result_loss  # 整体误差的求和
    print(running_loss)  # 每次循环会发现整体误差在不断减小
