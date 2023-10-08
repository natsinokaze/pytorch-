import torch
from torch import nn
from torch.nn import functional as F


class ResBlk1(nn.Module):
    #

    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk1, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()  # 如果ch_in等于ch_out则默认不做处理，否则将ch_in的值等于ch_out的值（维度匹配才能H(x)=x+F(x)）
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b,ch,h,3]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        # short cut [b,ch_out,h,w] add [b,ch_in,h,w]
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b,64,h,w] => [b,128,h,w]  注意h和w是随着ResBlk变换的
        self.blk1 = ResBlk1(64, 32)
        # [b,128,h,w] =>[b,256,h,w]
        self.blk2 = ResBlk1(32, 16)
        # [b,256,h,w] => [b,512,h,w]
        # self.blk3 = ResBlk(256, 512)
        # # [b,512,h,w] => [b,1024,h,w]
        # self.blk4 = ResBlk(512, 1024)

        self.outlayer1 = nn.Linear(16*32*32, 64)
        self.outlayer2 = nn.Linear(64, 10)


    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        #  [b,64,h,w] => [b,1024,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        # x = self.blk3(x)
        # x = self.blk4(x)

        x = x.view(x.size(0), -1)
        x = self.outlayer1(x)
        x = self.outlayer2(x)
        return x




model=ResNet18()

tmp = torch.randn(2,3, 32, 32)
out=model(tmp)

# print('renst:',out.shape)