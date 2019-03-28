import os
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import nn,optim
import numpy as np
from torch.nn import functional as F
from time import strftime, localtime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters

num_classes = 2
num_epochs = 5
batch_size = 10
learning_rate = 0.001
hidden_size = 100
num_workers = 0

check_file = 'checkpoints/my_net.pkl'
########data
test_data_root = 'data/test1/'
train_data_root = 'data/train/'

train_data = DogCat(train_data_root,train=True)
test_data = DogCat(test_data_root,train=False,test=True)
val_data = DogCat(train_data_root,train=False)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
val_dataloader = DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# from keras import layers,models
# mm = models.Sequential()
# mm.add(layers.Conv2D(32,(3,3),padding='VALID'))
# mm.add(layers.MaxPool2D())

class my_net(nn.Module):
    def __init__(self,num_classes=2):
        super(my_net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dp = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*7*7,512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,num_classes)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size()[0],-1)
        x = self.dp(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
####model
#model = models.ResNet34(num_classes=2).to(device)
model = my_net().to(device)
####loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



def val(model,dataloader):
    """
    计算模型在验证集上的信息
    """
    model.eval()########固定
    acc_val = 0
    total = 0
    loss_val = 0
    val_iteration = 0
    for i,(images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).sum().item() / len(labels)
        acc_val +=acc
        loss_val +=loss.item()
        val_iteration += 1

    acc_val /= val_iteration
    loss_val /= val_iteration

    model.train()####重启
    # print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    # print('loss of test images:%.3f' % (np.array(loss_my).mean()))
    return loss_val,acc_val


# Train the model
total_step = len(train_dataloader)

history={}
history['loss_train'] = []
history['loss_val'] = []
history['acc_train'] = []
history['acc_val'] = []

max_checks_without_progress = 30
checks_without_progress = 0
best_loss = np.infty

for epoch in range(50):
    loss_train = 0.0
    acc_train = 0.0
    iteration = 0
    for i, (images, labels) in enumerate(train_dataloader):
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
        # if (i + 1) % 10 == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},Acc: {:.3f}'
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),acc))
    ###evalutate
    loss_val,acc_val = val(model,val_dataloader)
    # #####提前停止
    if loss_val < best_loss:
        torch.save(model.state_dict(),check_file)
        best_loss = loss_val
        checks_without_progress = 0
    else:
        checks_without_progress += 1
        if checks_without_progress > max_checks_without_progress:
            print('early stopping!')
            break
    loss_train /= iteration
    acc_train /=iteration
    history['loss_train'].append(loss_train)
    history['acc_train'].append(acc_train)
    history['loss_val'].append(loss_val)
    history['acc_val'].append(acc_val)

    print('epoch: %d, train loss:%.3f, train_acc:%.3f,val loss:%.3f, val acc:%.3f'
          %(epoch+1,loss_train,acc_train,loss_val,acc_val))



import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(history['loss_train'],'b',label='loss_train')
plt.plot(history['acc_train'],'r',label='acc_train')
plt.plot(history['loss_val'],'g',label='loss_val')
plt.plot(history['acc_val'],'k',label='acc_val')
plt.legend()
plt.savefig('res.png')
plt.show()

# Test the model and submmit
# In test phase, we don't need to compute gradients (for memory efficiency)


model = my_net().to(device)
model.load_state_dict(torch.load(check_file))

model.eval()########固定
res_pro = []
res_id = []
for i,(images,path) in enumerate(test_dataloader):
    images = images.to(device)
    outputs = model(images)
    pro = F.softmax(outputs,dim=1)[:,1].detach().tolist()
    path = path.numpy().tolist()
    res_pro +=pro
    res_id +=path

#############保存结果
import pandas as pd

res = pd.DataFrame()
res['id'] = res_id
res['label'] = res_pro

res.to_csv('submit.txt',sep=' ',index=False)







    #
    # print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    # print('loss of test images:%.3f'%(np.array(loss_my).mean()))

res = []
probability = torch.nn.functional.softmax(outputs,dim=1)[:,0].detach().tolist()

batch_results = [(path_.item(),probability_) for path_,probability_ in zip(path,probability) ]



















