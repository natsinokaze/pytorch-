import torch
import torchvision
from torch.nn import MaxPool2d

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # MaxPool2d数据的类型不能是整形 因此要转为浮点型
# input = torch.reshape(input, (-1, 1, 5, 5))
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)

dataloader = DataLoader(dataset,batch_size=64)

class pool(torch.nn.Module):
    def __init__(self):
        super(pool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # ceil_mode=True表示当3*3的池化核遇到边界后仍然获取最大值并输出 False则不输出,
        # 默认stride=3,padding=0

    def forward(self, input):
        output = self.maxpool1(input)
        return output


maxpool = pool()

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input1", imgs, step)
    output = maxpool(imgs)

    writer.add_images("output1", output, step)
    step += 1

writer.close()