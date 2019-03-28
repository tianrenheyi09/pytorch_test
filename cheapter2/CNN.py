import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters

num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
hidden_size = 100

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

####
class ConvNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc1 = nn.Linear(7*7*32,hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size()[0],-1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = ConvNet(num_classes).to(device)


# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
import time
total_step = len(train_loader)
for epoch in range(num_epochs):
    loss_train = 0.0
    acc_train = 0.0
    iteration = 0
    t1 = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
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
    ut = time.time()-t1
    print('time:%.3f,epoch: %d, train loss:%.3f, train_acc:%.3f'%(ut,epoch+1,loss_train,acc_train))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
import numpy as np
model.eval()
with torch.no_grad():
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss_my .append(loss.item())
        acc.append((predicted==labels).cpu().numpy().mean())
        iteration +=1

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    print('loss of test images:%.3f'%(np.array(loss_my).mean()))


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(loss_my,'b',linewidth=2,label='loss_test')
plt.plot(acc,'r',linewidth=2,label='acc')
plt.show()

from PIL import Image
img = Image.open('1.png')
plt.figure('hua')
plt.imshow(img)
plt.show()
plt.axis('off')


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')