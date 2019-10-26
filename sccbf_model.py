import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import resnet_model
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.proj0 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)

        self.proj1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)

        self.proj2 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)

        self.fc_concat = torch.nn.Linear(512*512*3+512*512*3, 200)

        self.softmax = nn.LogSoftmax(dim=1)
        self.avgpool = nn.AvgPool2d(kernel_size=14)

        self.features = resnet_model.resnet50(pretrained=True,
                                              model_root='D:/Zhen/LMX/LMX/resnet50.pth')


    def forward(self, x):
        batch_size = x.size(0)
        feature4_0_o, feature4_1_o, feature4_2_o = self.features(x)

        feature4_0 = self.proj0(feature4_0_o)

        feature4_1 = self.proj1(feature4_1_o)

        feature4_2 = self.proj2(feature4_2_o)

        feature4_0 = feature4_0.view(batch_size, feature4_0.size()[1], -1)   
        feature4_0_1_t = feature4_0.permute(0,2,1)   
        feature4_0_t = torch.matmul(feature4_0, feature4_0_1_t) 
        feature4_0_t = feature4_0_t.view(batch_size, -1)   

        feature4_1 = feature4_1.view(batch_size, feature4_1.size()[1], -1)
        feature4_1_1_t = feature4_1.permute(0, 2, 1)
        feature4_1_t = torch.matmul(feature4_1, feature4_1_1_t)
        feature4_1_t = feature4_1_t.view(batch_size, -1)

        feature4_2 = feature4_2.view(batch_size, feature4_2.size()[1], -1)
        feature4_2_1_t = feature4_2.permute(0, 2, 1)
        feature4_2_t = torch.matmul(feature4_2, feature4_2_1_t)
        feature4_2_t = feature4_2_t.view(batch_size, -1)

        inter1 = torch.matmul(feature4_0, feature4_1_1_t)
        inter2 = torch.matmul(feature4_0, feature4_2_1_t)
        inter3 = torch.matmul(feature4_1, feature4_2_1_t)

        inter1 = inter1.view(batch_size, -1)
        inter2 = inter2.view(batch_size, -1)
        inter3 = inter3.view(batch_size, -1)

        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

        feature4_0_1 = torch.nn.functional.normalize(torch.sign(feature4_0_t) * torch.sqrt(torch.abs(feature4_0_t) + 1e-10))
        feature4_1_1 = torch.nn.functional.normalize(torch.sign(feature4_1_t) * torch.sqrt(torch.abs(feature4_1_t) + 1e-10))
        feature4_2_1 = torch.nn.functional.normalize(torch.sign(feature4_2_t) * torch.sqrt(torch.abs(feature4_2_t) + 1e-10))

        result = torch.cat((result1, result2, result3, feature4_0_1, feature4_1_1, feature4_2_1), 1)  

        result = self.fc_concat(result)
        return self.softmax(result)
