from d2l import torch as d2l
from torch import nn
import torch.utils.data

import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from liyimengPJ1.formal.qtTest.MyDataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
 # 根据您的数据集定义自己的数据加载器

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
makedir('./checkpoints/LGE')
# 加载预训练模型，这里假设您的模型名为 MyModel，并保存在 model.pth 文件中
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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


model = MyModel()

# 加载预训练模型
pretrained_dict = torch.load('D:\liyimengPJ\liyimengPJ1\LITTLE\checkpoints\C0\model-best_C0.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 冻结所有层的参数
# for param in net[:9].parameters():
for param in model.model1[:12].parameters():
    param.requires_grad = False

# 定义新的分类器，替换原模型的最后一层
# num_ftrs = model.model1[-1].in_features
# model.model1[-1] = nn.Linear(num_ftrs, num_classes)

# 将模型移动到 GPU 上进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

root1 = r'D:\PJ1data\apre\LGE_train'
root2 = r'D:\PJ1data\apre\LGE_test'

# 加载数据集并获取 DataLoader，这里假设您已经定义了自己的数据加载器 MyDataset 和 MyDataLoader
train_dataset = MyDataset(root=root1, mode='train', truncation=True)
train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = MyDataset(root=root2, mode='test', truncation=True)
test_iter = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
# 训练模型
timer, num_batches = d2l.Timer(), len(train_iter)
writer = SummaryWriter("../logs_train_LGE_pre")
best_acc = 0
for epoch in range(num_epochs):
    print(f'epoch:{epoch}')
    metric = d2l.Accumulator(3)
    model.train()
    for i, (img, label) in tqdm(enumerate(train_iter)):   #tqdm函数的用法
        timer.start()
        optimizer.zero_grad()
        img, label= img.to(device), label.to(device)

        label_pre = model(img)
        los = criterion(label_pre, label)
        los.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(los * img.shape[0], d2l.accuracy(label_pre, label), img.shape[0])
        timer.stop()
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_loss, epoch + (i + 1) / num_batches)
            writer.add_scalar('train accurary', train_acc, epoch + (i + 1) / num_batches)
    test_acc = d2l.evaluate_accuracy_gpu(model, test_iter)
    writer.add_scalar('test accurary', test_acc, epoch + 1)

    if test_acc > best_acc:
        torch.save(model.state_dict(), "./checkpoints/LGE/model-best_LGE_pre.pth")
        best_acc = test_acc
    torch.save(model.state_dict(), "./checkpoints/LGE/model-best_LGE_pre.pth")
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
      f'test acc {test_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')
writer.close()

# 微调
def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
makedir('./checkpoints/LGE')
# 加载预训练模型，这里假设您的模型名为 MyModel，并保存在 model.pth 文件中
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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


model = MyModel()

# 加载预训练模型
pretrained_dict = torch.load('./checkpoints/LGE/model-best_LGE_pre.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 冻结所有层的参数
# for param in net[:9].parameters():
for param in model.model1[:12].parameters():
    param.requires_grad = True

# 定义新的分类器，替换原模型的最后一层
# num_ftrs = model.model1[-1].in_features
# model.model1[-1] = nn.Linear(num_ftrs, num_classes)

# 将模型移动到 GPU 上进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

root1 = r'D:\PJ1data\apre\LGE_train'
root2 = r'D:\PJ1data\apre\LGE_test'
train_dataset = MyDataset(root=root1, mode='train', truncation=True)
train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = MyDataset(root=root2, mode='test', truncation=True)
test_iter = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
# 训练模型
timer, num_batches = d2l.Timer(), len(train_iter)
writer = SummaryWriter("../logs_train_LGE_last")
best_acc = 0
for epoch in range(num_epochs):
    print(f'epoch:{epoch}')
    metric = d2l.Accumulator(3)
    model.train()
    for i, (img, label) in tqdm(enumerate(train_iter)):   #tqdm函数的用法
        timer.start()
        optimizer.zero_grad()
        img, label= img.to(device), label.to(device)

        label_pre = model(img)
        los = criterion(label_pre, label)
        los.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(los * img.shape[0], d2l.accuracy(label_pre, label), img.shape[0])
        timer.stop()
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_loss, epoch + (i + 1) / num_batches)
            writer.add_scalar('train accurary', train_acc, epoch + (i + 1) / num_batches)
    test_acc = d2l.evaluate_accuracy_gpu(model, test_iter)
    writer.add_scalar('test accurary', test_acc, epoch + 1)

    if test_acc > best_acc:
        torch.save(model.state_dict(), "./checkpoints/LGE/model-best_LGE_last.pth")
        best_acc = test_acc
    torch.save(model.state_dict(), "./checkpoints/LGE/model-best_LGE_last.pth")
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
      f'test acc {test_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')
writer.close()
