# 准备数据集
import time

import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from liyimengPJ1.formal.qtTest.MyDataset import MyDataset
root1 = r'D:\liyimengPJ\liyimengPJ1\data\C0\train'
root2 = r'D:\liyimengPJ\liyimengPJ1\data\C0\test'
num_workers = 0

train_dataset = MyDataset(root=root1, mode='train', truncation=True)
test_dataset = MyDataset(root=root2, mode='test', truncation=True)

# 利用Dataloader加载数据集
train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
# 创建网络模型
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
net = Net()
net = net.cuda()
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
# 损失函数
net.apply(init_weights)
device = d2l.try_gpu(0)
print('training on', device)
net.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网路参数
total_train_step = 0
total_test_step = 0
epoch = 40

writer = SummaryWriter("../logs_train")
start_time = time.time()

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    for data in train_iter:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            end_time = time.time()
            print(end_time - start_time, 's')
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))

            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_iter:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/len(train_dataset)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_dataset), total_test_step)
    total_test_step = total_test_step + 1
    torch.save(net, "C0_{}.pth".format(i))
    print("模型已保存")
writer.close()
