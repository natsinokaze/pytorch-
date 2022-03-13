import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Seq(nn.Module):
    def __init__(self):
        super(Seq, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2) # 3@32*32 --> 32@32*32
        # self.maxpool1 = MaxPool2d(2) # 32@32*32 --> 32@16*16
        # self.conv2 = Conv2d(32, 32, 5, padding=2) # 32@16*16 --> 32@16*16
        # self.maxpool2 = MaxPool2d(2) # 32@16*16 --> 32@8*8
        # self.conv3 = Conv2d(32, 64, 5, padding=2) # 32@8*8 --> 64@8*8
        # self.maxpool3 = MaxPool2d(2) # 64@8*8 --> 64@4*4
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64) # 第一项 一定要正确这里是1024的原因是经过maxpool3后有64*4*4=1024个元素
        # self.linear2 = Linear(64, 10)

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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

seq = Seq()

input = torch.ones((64, 3, 32, 32)) # 创建和输入数据同样大小的值全为1的tensor数组 用于对模型进行检查
output = seq(input)
print(output.shape) # 能输出结果则说明模型没有问题

writer = SummaryWriter("logs/logs_seq")
writer.add_graph(Seq(), input) # 用于显示流程 即将seq(input)的流程进行显示
writer.close()