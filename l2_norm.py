import torch

from lib import *

class L2Norm(nn.Module):
    def __init__(self, input_channels = 512, scale = 20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameter()
        self.esp =1e-10

    def reset_parameter(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        # x.size() = (batch_size, channels, height, weight)
        # L2Norm
        # print(x.size())
        norm = x.pow(2).sum(dim=1, keepdims=True).sqrt() + self.esp # tính nort theo chiều channels, size = (batch_size, 1, height, weight)
        # print(norm.size())
        x = torch.div(x, norm) #(batch_size, channels, height, weight)
        # print(x)
        # weight.size() = 512
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print(weights)
        return weights*x # scale x bằng cách nhân với weights( size = (batch_size, channels, height, weight) với các channels có giá trị = scale(=20))
if __name__ =="__main__":
    input = torch.randn(4, 512, 19, 19)
    l2n = L2Norm()
    a= l2n.forward(input)
    print("đây là a: ",a)