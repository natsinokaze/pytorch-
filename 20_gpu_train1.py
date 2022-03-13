import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 要使用GPU训练，需要在网络模型、数据（输入，标注），损失函数三处添加.cuda()
# 搭建神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


train_data = torchvision.datasets.CIFAR10(root="./dataset", transform=torchvision.transforms.ToTensor(), train=True,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", transform=torchvision.transforms.ToTensor(), train=False,
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data_size, test_data_size)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
cnn = CNN()
if torch.cuda.is_available():  # 严格意义上每个cuda前都应该添加这句判断
    cnn = cnn.cuda()  # 除了imgs和targets，其他的都可以直接写成 *.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)  # 随机梯度下降

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter("logs/logs_train")

for i in range(epoch):
    print("---------第{}轮训练开始--------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = cnn(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:  # 每100次报一次
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))  # 加上item可以为后续的loss可视化作基础
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0  # 用于存储每回合的总的loss的值
    total_accurary = 0  # 用于存储准确率

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = cnn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            accuray = (outputs.argmax(1) == targets).sum()  # 表示横向获取值最大的索引 如果等于targets的值则为true（值为1），获取所有true的总个数
            total_accurary += accuray

    print("本回合整体上的LOSS：{}".format(total_test_loss))
    print("本回合整体上的正确率：{}".format(total_accurary / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("tedt_accurary", total_accurary / test_data_size, total_test_step)
    total_test_step += 1

    # 保存每回合的模型
    torch.save(cnn, "save/model_save_{}.pth".format(i))
writer.close()
