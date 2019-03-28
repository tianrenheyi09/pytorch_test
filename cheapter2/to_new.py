import numpy as np
import torch as t
from __future__ import print_function

# t.__version__
#
# x = t.Tensor(5,3)
# x = t.Tensor([[1,2],[2,3]])
#
# x = t.rand(5,3)
#
# print(x.size()[0],'shape',x.size()[1])
#
# y = t.rand(5,3)
#
# x+y
# t.add(x,y)
#
# result = t.Tensor(5,3)
# t.add(x,y,out = result)
# result
#
# print('first +++')
# y.add(x)
# print(y)
# print('second +++')
# y.add_(x)
# print(y)
#
# a = t.ones(5)
#
# b = a.numpy()
# b
#
# a = np.ones(5)
# b = t.from_numpy(a)
# print(a)
# print(b)
# ####a和b共享内存
# b.add_(1)
# print(a);print(b)
#
# #########取tensor中的元素
# scalar = b[0:2]
# scalar.size()
# scalar1 = b[0]
# scalar1.size()
# # scalar.item()##########scalar.item只能用于一个元素
# scalar1.item()
#
# tensor = t.tensor([3,4])
# old_tensor = tensor
# new_tensor = t.tensor(old_tensor)
# new_tensor[0] = 1111
# old_tensor, new_tensor
#
# new_tensor = old_tensor.detach()
# new_tensor[0] = 1111
# old_tensor, new_tensor
#
# # 在不支持CUDA的机器下，下一步还是在CPU上运行
# device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# x = x.to(device)
# y = y.to(device)
# z = x+y
#
# ###########
#
x = t.ones(2,2,requires_grad=True)
y = x.sum()
y.backward()
#
# y.grad_fn
# y.backward()
# x.grad
#
# y.backward()
# x.grad

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):

        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
net = Net()
print(net)

params = list(net.parameters())

for name,parameters in net.named_parameters():
    print(name,':',parameters.size())

input = t.randn(1,1,32,32)
out = net(input)
out.size()

net.zero_grad()
out.backward(t.ones(1,10))

output = net(input)
target =Variable(t.range(1, 10))
criterion = nn.MSELoss()
loss = criterion(output,target)
loss



optimizer = optim.SGD(net.parameters(),lr = 0.01)
############训练过程中先进行梯度清零
output = net(input)
loss = criterion(output,target)

loss.backward()

optimizer.step()
'''
###############################
#############CIFAR-10数据加载以及处理
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image

show = ToPILImage()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = tv.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform
)


trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size = 4,
    shuffle=True,
    num_workers=2
)
testset = tv.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)
testloader = t.utils.data.DataLoader(
    testset,
    batch_size = 4,
    shuffle=False,
    num_workers=2
)

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# from data_load import load_CIFAR10
# filepath = 'data/cifar-10-batches-py'
# X_train,y_train,X_test,y_test = load_CIFAR10(filepath)
#
# print('train data shape:',X_train.shape)
# print('train labels shape:',y_train.shape)
# print('test data shape:',X_test.shape)
# print('test labels shape:',y_test.shape)

(data,label) = trainset[100]
print(classes[label])

Image._show(show((data+1)/2).resize((100,100)))

dataiter = iter(trainloader)
images, labels = dataiter.next() # 返回4张图片及标签
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))

Image._show(show(tv.utils.make_grid((images+1)/2)).resize((400,100)))

# from PIL import Image
# img = Image.open("1.png")
# img.show()


class Net(nn.Module):
    def __init__(self):

        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
#########定义损失函数和优化器
from torch import optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
##########训练网络
###输入数据，前向——后向。调整参数

t.set_num_threads(8)

for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        ###梯度清零
        optimizer.zero_grad()
        ###前向+后向
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        ####更新参数
        optimizer.step()
        ####打印log信息
        running_loss + loss.item()
        if i%2000 ==1999:
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
print('finished')

#####测试一下
dataiter = iter(testloader)
images,labels = dataiter.__next__()
print('实际label',' '.join(classes[labels[j]] for j in range(4)))

Image._show(show(tv.utils.make_grid(images/2-0.5)).resize((400,100)))
###网络的预测输出
outputs = net(images)
_,predicted = t.max(outputs.data,1)
print('预测结果:',' '.join(classes[predicted[j]] for j in range(4)))

#########整个测试集上
correct = 0.0
total = 0

with t.no_grad():
    for data in testloader:
        images,labels = data
        outputs = net(images)
        _,predicted = t.max(outputs,1)
        total += labels.size()[0]
        correct += (predicted==labels).sum()

print('10000张测试集准确率:%d%%'%(100*correct/total))

