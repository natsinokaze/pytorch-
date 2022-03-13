import os
from torch.utils.data import Dataset
from PIL import Image

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 获取所有图片的路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取所有图片的名称
        self.img_name=os.listdir(self.path)

    def __getitem__(self, idx):
        # 获取某一个图片
        img_name = self.img_name[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        # 获取列表长度
        return len(self.img_path)

root_dir="C:\\Users\\XIAO\\learn\\pycharm\\hymenoptera_data\\train"
ants_label_dir="ants"
ants_dataset=MyData(root_dir, ants_label_dir)
a=ants_dataset.__getitem__(0)#等价于a=ants_dataset[0] 它包含了两个值，一个图片 一个标签
img, label=a
img.show()