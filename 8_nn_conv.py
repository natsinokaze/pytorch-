import torch
import torch.nn.functional as F
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])# 设置卷积核

input = torch.reshape(input, (1, 1, 5, 5))# 因为conv2d要求输入的input有四个参数  input tensor of shape (minibatch,in_channels,iH,iW)
kernel = torch.reshape(kernel, (1, 1, 3, 3))
output1 = F.conv2d(input, kernel, stride = 1)# padding默认设置为0
print(output1)
output2 = F.conv2d(input, kernel, stride = [1,2])# stride为设置步长右走1步下走两步 stride = [1,1]等价于stride = 1
print(output2)
output3 = F.conv2d(input, kernel, stride = 1,padding=[1, 1])# 若padding为1个参数则表示四周填充相同行列 padding=[1,1]等价于padding=1
print(output3)