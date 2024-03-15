import os
import random
import numpy as np
import torch
from torchvision import transforms
import pickle
import cv2

"""
将自己的数据集转码为二进制格式，转码后能极大程度提高数据集的读取效率。
导出的二进制文件为一个字典，键分别有'data'，'labels'，'filenames'
此代码针对的原始数据集格式为每一个文件夹表示一个类，每一个文件夹中的图片均为其所在文件夹的名称
"""

datasets_path=r'D:\deep_learning\datasets\NWPU-RESISC45\original'
class_name=os.listdir(r'D:\deep_learning\datasets\NWPU-RESISC45\original')

train_ratio=0.8
test_ratio=1-train_ratio

#创建训练集和测试集
train_information_data=[]
test_information_data=[]

class_dict={}
train_file_name=[]
test_file_name=[]

for index, classes in enumerate(class_name):
    class_dir=os.path.join(datasets_path,classes)
    images_dir=os.listdir(class_dir)

    random.shuffle(images_dir) # 某个类的样本顺序打乱

    num_train_samples=int(len(images_dir)* train_ratio) # 计算某个类的训练样本数量
    train_samples = images_dir[:num_train_samples]
    test_samples = images_dir[num_train_samples:]

    train_information_data.extend([(os.path.join(class_dir, x), classes) for x in train_samples])
    test_information_data.extend([(os.path.join(class_dir, x), classes) for x in test_samples])
    train_file_name.extend(train_samples)
    test_file_name.extend(test_samples)

    class_dict.update({classes:index})

print(class_dict)



data=cv2.imread(train_information_data[0][0])
data_shape=data.shape
train_num=len(train_information_data)
test_num=len(test_information_data)
train_label_index=[]
test_label_index=[]
train_data=np.zeros((train_num,*data_shape),dtype=np.uint8)
test_data=np.zeros((test_num,*data_shape),dtype=np.uint8)

transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(224)]) # 数据预处理




# train_data=[]
# test_data=[]
'''将训练集和测试集的数据、标签分别存在一个变量中'''
for index, (images_dir,type) in enumerate(test_information_data):
    data=cv2.imread(images_dir)
    test_data[index] = data
    test_label_index.append(class_dict[type])

for index, (images_dir,type) in enumerate(train_information_data):
    data=cv2.imread(images_dir)
    train_data[index] = data
    train_label_index.append(class_dict[type])

'''进行转码'''
with open(r'D:\deep_learning\datasets\NWPU-RESISC45\pickle\test_data.bin', 'wb') as f:
    pickle.dump(({'data':test_data,'labels':test_label_index,'filenames':test_file_name}), f)

with open(r'D:\deep_learning\datasets\NWPU-RESISC45\pickle\train_data.bin', 'wb') as f:
    pickle.dump({'data':train_data,'labels':train_label_index,'filenames':train_file_name}, f)

'''后续要读取数据的格式'''
# with open(r'D:\deep_learning\test_data.bin', 'rb') as f:
#     loaded_data = pickle.load(f, encoding='bytes') #最好加上encoding='bytes'（除非报错）
# print(loaded_data['labels'][10000],loaded_data['filenames'][10000]) #验证是否正确
print('a')

