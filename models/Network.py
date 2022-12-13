import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self, n_outputs):
        super(Net, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        # print('net', data_set)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, n_outputs)  # n_outputs 输出类别

    def forward(self, x):
        # 28 * 28
        # print(x.size())
        x = self.conv1(x)
        # 26 * 26
        x = F.relu(x)
        x = self.conv2(x)
        # 24 * 24
        x = F.max_pool2d(x, 2)  # 池化
        # 12 * 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # 9216 * 1
        # print(x.size())
        x = self.fc1(x)
        # 128 * 1
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # 10 * 1
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x




