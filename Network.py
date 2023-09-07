# Python 3.10.9
# 本程序中的类DualInputNet用于生成分类模型架构
import torch
from torch import nn
from transformers import ResNetConfig, ResNetModel

# configuration = ResNetConfig(num_channels=1, hidden_sizes=[64, 128, 256, 512], layer_type='basic')
# model = ResNetModel(configuration)


class DualInputNet(nn.Module):
    def __init__(self, cnn1=None, cnn2=None):
        super(DualInputNet, self).__init__()
        if not cnn1 and not cnn2:
            configuration = ResNetConfig(num_channels=1, hidden_sizes=[64, 128, 256, 512], layer_type='basic')
            self.cnn1 = ResNetModel(configuration)
            self.cnn1.from_pretrained("microsoft/resnet-18")
            self.cnn2 = ResNetModel(configuration)
            self.cnn2.from_pretrained("microsoft/resnet-18")
        else:
            self.cnn1 = cnn1
            self.cnn2 = cnn2
        self.flat = nn.Flatten()
        # self.combination = nn.Linear(1024, 2)
        self.combination = nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=1))
        # self.combination = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 2)
        # )

    def forward(self, x1, x2):
        x1 = self.flat(self.cnn1(x1).pooler_output)
        x2 = self.flat(self.cnn2(x2).pooler_output)
        x = torch.concat([x1, x2], dim=1)
        return self.combination(x)


class SingleInputNet(nn.Module):
    def __init__(self, cnn1=None):
        super(SingleInputNet, self).__init__()
        if not cnn1:
            configuration = ResNetConfig(num_channels=1, hidden_sizes=[64, 128, 256, 512], layer_type='basic')
            self.cnn1 = ResNetModel(configuration)
            self.cnn1.from_pretrained("microsoft/resnet-18")
        else:
            self.cnn1 = cnn1
        self.flat = nn.Flatten()
        self.combination = nn.Sequential(nn.Linear(512, 2), nn.Softmax(dim=1))

    def forward(self, x):
        # 单通道网络输入只有一张图像
        x = self.flat(self.cnn1(x).pooler_output)
        return self.combination(x)
