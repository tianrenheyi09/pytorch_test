from torch import nn
import torch as t
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
# Hyper-parameters

num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
hidden_size = 100

#######Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

#  dataset (images and labels)
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = t.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = t.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)




class ResidualBlock(nn.Module):
    def __init__(self,in_channel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel,outchannel,3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ResNet,self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)

        )

        self.layer1 = self._make_layer(16,16,2)
        self.layer2 = self._make_layer(16,32,2,stride=2)
        self.layer3 = self._make_layer(32,64,2,stride=2)
        self.avg_pool = nn.AvgPool2d(8)


        # self.ave = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(64,num_classes)


    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x



model =ResNet(num_classes).to(device)
# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    loss_train = 0.0
    acc_train = 0.0
    iteration = 0
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = t.max(outputs.data, 1)
        acc = (predicted==labels).sum().item()/len(labels)

        loss_train +=loss.item()
        acc_train +=acc
        iteration +=1

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},Acc: {:.3f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),acc))

    loss_train /= iteration
    acc_train /=iteration
    print('epoch: %d, train loss:%.3f, train_acc:%.3f'%(epoch+1,loss_train,acc_train))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
import numpy as np
model.eval()
with t.no_grad():
    correct = 0
    total = 0
    loss_my = []
    iteration = 0
    acc = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss_my .append(loss.item())
        acc.append((predicted==labels).sum().item()/len(labels))
        iteration +=1

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    print('loss of test images:%.3f'%(np.array(loss_my).mean()))


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(loss_my,'b',linewidth=2,label='loss_test')
plt.plot(acc,'r',linewidth=2,label='acc')
plt.show()


# Save the model checkpoint
t.save(model.state_dict(), 'model.ckpt')


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,6))

x = np.linspace(0,5,500)
y = np.sin(x)
plt.plot(x,y)