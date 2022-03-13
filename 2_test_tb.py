from torch.utils.tensorboard import SummaryWriter
import numpy as np # 用于将PIL类型转换为add_image 函数所需要的numpy型
from PIL import Image

writer = SummaryWriter("logs")
img_path = "hymenoptera_data/train/ants/0013035.jpg"
img_pil = Image.open(img_path)
img_np=np.array(img_pil) # 将PIL类型转换为add_image 函数所需要的numpy型
writer.add_image("title", img_np, 1, dataformats='HWC')# 1表示步长 及显示多少个图 dataformats是根据不同的数据类型设置 三个字母要分别对应数据的长、宽、通道数
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
writer.close()
print("finish")
