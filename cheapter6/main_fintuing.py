import os
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import nn,optim
import numpy as np
from torch.nn import functional as F
import torchvision.models as models
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

####model
model = models.resnet34(pretrained=True)
# model = models.resnet18(pretrained=True)
# model = models.resnet50(pretrained=True)
#
# model = models.vgg16(pretrained=True)
# model = models.vgg11(pretrained=True)
# model = models.vgg13(pretrained=True)
######参数修改
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(512,256),
    nn.ReLU(inplace=True),
    nn.Linear(256,2)
)

for param in model.parameters():
    print(param.requires_grad)

####loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)



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
print('开始训练模型')
total_step = len(train_dataloader)

history={}
history['loss_train'] = []
history['loss_val'] = []
history['acc_train'] = []
history['acc_val'] = []

max_checks_without_progress = 5
checks_without_progress = 0
best_loss = np.infty

for epoch in range(3):
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
    #####提前停止
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


print('显示训练集合验证集的loss和acc')
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

#####################加载训练过程中的最优参数

model = models.resnet34(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(512,256),
    nn.ReLU(inplace=True),
    nn.Linear(256,2)
)
model.load_state_dict(torch.load(check_file))

print("开始测试模型")
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
print("保存结果")
import pandas as pd

res = pd.DataFrame()
res['id'] = res_id
res['label'] = res_pro

res.to_csv('submit.txt',sep=' ',index=False)

##########################显示其中的图片以及label
print("开始显示测试图片")
from PIL import Image
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
kind={}
kind[1] = 'dog'
kind[0] = 'cat'
for i in range(1,9):
    path = test_data_root+str(i)+'.jpg'
    plt.subplot(4,2,i)
    img = Image.open(path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(kind[res['label'][i-1]>0.5])





    #
    # print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    # print('loss of test images:%.3f'%(np.array(loss_my).mean()))





















