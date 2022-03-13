import torch
from torch.nn import L1Loss
from torch import nn
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
target = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum') # 默认为mean:各项之差的绝对值的和除以元素个数
result = loss(inputs, targets)
print(result)

loss = nn.MSELoss() # 计算平方差再除以元素个数 (0+0+2^2)/3
result = loss(inputs, targets)
print(result)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1]) # 表示对应x的索引为1 即 target=1的概率为0.2
x = torch.reshape(x, (1, 3)) # 因为loss_cross 要求x为 (batch_size,class)，这里batch_size为1，分类为3，对y没有要求
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y) # -0.2 + ln(exp(0.1)+exp(0.2)+exp(0.3))=1.019
print(result_cross)