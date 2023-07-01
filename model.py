import os
import sys
import numpy as np
import torch
from SimpleITK import ReadImage, GetArrayFromImage
from skimage import exposure
from torch import nn
from torch.nn import Conv2d
from torchvision import transforms
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(".."), relative_path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model1 = nn.Sequential(
            Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

class utilss(Net):
    def __init__(self, root=None, mode='train', truncation=False):

        super(Net, self).__init__()
        # 预处理nii图像，切片，直方图均衡化后为四维数组
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),

            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

    def process_img(self, img):
        nii_file = ReadImage(img)
        nii_data = GetArrayFromImage(nii_file)
        nii_data = np.transpose(nii_data, (1, 2, 0))
        # 沿着最小维度切片
        self.n_slices = nii_data.shape[2]
        data_list = []
        data = []
        for i in range(self.n_slices):
            slice_3d = nii_data[..., i]
            # 直方图均衡化
            slice_3d_eq = exposure.equalize_hist(slice_3d) * 255
            # 第一组阶段
            slice_3d_clip1 = np.where((slice_3d_eq >= 0.6 * slice_3d_eq.max()), 0.6 * slice_3d_eq.max(), slice_3d_eq)
            data.append(slice_3d_clip1)
            # 第二组阶段
            slice_3d_clip2 = np.where((slice_3d_eq >= 0.8 * slice_3d_eq.max()), 0.8 * slice_3d_eq.max(), slice_3d_eq)
            data.append(slice_3d_clip2)
            # 第三组阶段
            slice_3d_clip3 = np.where((slice_3d_eq >= 1.0 * slice_3d_eq.max()), 1.0 * slice_3d_eq.max(), slice_3d_eq)
            data.append(slice_3d_clip3)
            data_list.append(data)
            data = []
        # 构建四维数组
        data_array = np.array(data_list)
        return data_array

    def predict(self, img):
        self.pre_predict = self.process_img(img)
        self.net_C0 = Net()
        self.net_C0.load_state_dict(torch.load("D:\liyimengPJ\liyimengPJ1\LITTLE\checkpoints\C0\model-best_C0.pth"))
        self.net_C0.eval()
        self.net_LGE = Net()
        self.net_LGE.load_state_dict(
            torch.load("D:\liyimengPJ\liyimengPJ1\LITTLE\checkpoints\LGE\model-best_LGE_last.pth"))
        self.net_LGE.eval()
        self.net_T2 = Net()
        self.net_T2.load_state_dict(
            torch.load("D:\liyimengPJ\liyimengPJ1\LITTLE\checkpoints\T2\model-best_T2_last.pth"))
        self.net_T2.eval()

        if "C0" in img:
            self.model = self.net_C0
        if "LGE" in img:
            self.model = self.net_LGE
        if "T2" in img:
            self.model = self.net_T2
        resultDic = dict()
        for i in range(self.n_slices):
            data_sli = torch.tensor(self.pre_predict[i - 1]).float()
            data_sli = self.data_transforms['test'](data_sli)
            data_sli = data_sli.reshape(1, 3, 256, 256)
            with torch.no_grad():
                # 将测试数据输入到模型中进行预测
                pre_predictions = self.model(data_sli)
                key = pre_predictions.argmax()
                key = key.item()
                if key in resultDic.keys():
                    resultDic[key] += 1
                else:
                    resultDic[key] = 1
        max_count = max(resultDic.values())
        self.direction = [k for k, v in resultDic.items() if v == max_count]
        self.direction = int(self.direction[0])
        index_to_label = ['100', '101', '110', '000', '001', '010', '011', '111']
        label = index_to_label[self.direction]
        return (label)








