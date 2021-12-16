from lib import *

# pool of square window of size=3, stride=2
# m = nn.MaxPool2d(3, stride=1, padding=1)
#  # pool of non-square window
# # m = nn.MaxPool2d((3, 2), stride=(2, 1))
# input = torch.randn(1, 19, 19)
# output = m(input)
# print("input: \n",input.shape)
# print("output: \n",output.shape)
m = nn.Conv2d(2, 28, 3,2,1)

input = torch.randn(20, 2, 5, 5)

output = m(input)

print(input.shape)
print(output.shape)
