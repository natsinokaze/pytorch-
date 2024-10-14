import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

save_dir=r'G:\deep_learning\test\save_model'

'''
神经网络进行非线性拟合的例子，两个输入特征去拟合有个倒立的螺旋函数
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成螺旋参数
theta = np.linspace(0, 8 * np.pi, 1000)  # 角度，从0到8π，使螺旋上升多圈
z_data = np.linspace(0, 10, 1000)  # z轴，从0到10，表示高度
r_data = z_data / 2  # 半径随着z轴增大而增加，形成圆锥

# 添加噪声
r_data += np.random.normal(0, 0.1, r_data.shape)  # 对半径添加噪声
z_data += np.random.normal(0, 0.1, z_data.shape)  # 对z轴添加噪声

# 将极坐标转换为直角坐标
x_data = r_data * np.cos(theta)
y_data = r_data * np.sin(theta)

# 将 x_data 和 y_data 合并成输入矩阵，z_data 为目标输出
X_train = np.column_stack((x_data, y_data))
y_train = z_data

# 创建3D图形
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制螺旋线
# ax.plot(x_data, y_data, z_data, color='blue', lw=2)
#
# # 设置轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 设置标题
# ax.set_title('Noisy Spiral on a Conical Surface')
#
# # 显示图形
# plt.show()



# 转换为 PyTorch Tensor，并调整形状
X_train = torch.tensor(X_train, dtype=torch.float32)  # 输入是二维的 (1000, 2)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 输出是 (1000, 1)

# 定义模型
class TemperatureRegressor(nn.Module):
     def __init__(self):
         super(TemperatureRegressor, self).__init__()
         self.fc1 = nn.Linear(2, 128)
         self.fc2 = nn.Linear(128, 64)
         self.fc3 = nn.Linear(64, 1)
         self.Tanh = nn.Tanh()
         self.Prelu = nn.PReLU()

     def forward(self, x):
         x = self.Tanh(self.fc1(x))
         x = self.Tanh(self.fc2(x))
         x = self.fc3(x)
         return x

# 实例化模型、损失函数和优化器
model = TemperatureRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 150000
for epoch in range(epochs):
    model.train()

    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:   # 每1000个epoch打印一次loss
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    if epoch % 50000 == 0:
        torch.save(model, os.path.join(save_dir, f'model_{epoch}.pth'))

model.eval()
with torch.no_grad():
    predicted_z = model(torch.tensor(np.column_stack((x_data, y_data)), dtype=torch.float32)).numpy()
    predicted_z = predicted_z.flatten()
# 创建3D图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据
ax.plot(x_data, y_data, z_data, color='blue', lw=2, label='Actual Data')

# 绘制预测数据
ax.plot(x_data, y_data, predicted_z, color='red', lw=1, label='Predicted Data')

# 设置轴标签
ax.set_xlabel('X Data')
ax.set_ylabel('Y Data')
ax.set_zlabel('Z Data')

# 设置标题
ax.set_title('3D Visualization of Actual and Predicted Data')

# 显示图例
ax.legend()

# 显示图形
plt.show()

