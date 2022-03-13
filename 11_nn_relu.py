import torch
import torchvision
from torch.nn import MaxPool2d, ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)

dataloader = DataLoader(dataset,batch_size=64)

class pool(torch.nn.Module):
    def __init__(self):
        super(pool, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


maxpool = pool()

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input2", imgs, step)
    output = maxpool(imgs)

    writer.add_images("output2", output, step)
    step += 1

writer.close()