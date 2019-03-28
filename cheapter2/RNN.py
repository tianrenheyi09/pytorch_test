import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #
        # # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, (h_n, h_c) = self.lstm(x, None)  # # None represents zero initial hidden state


        out = self.fc(out[:,-1,:])
        return out

model = RNN(input_size,hidden_size,num_layers,num_classes)


# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#########不能用optim.SGD


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    loss_train = 0.0
    acc_train = 0.0
    iteration = 0
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1,sequence_length,input_size).to(device)
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
    print('epoch: %d, train loss:%.3f, train_acc:%.3f'%(epoch+1,loss_train,acc_train))


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
        images = images.reshape(-1,sequence_length,input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss_my .append(loss.item())
        acc.append((predicted==labels).numpy().mean())
        iteration +=1

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    print('loss of test images:%.3f'%(np.array(loss_my).mean()))


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(loss_my,'b',linewidth=2,label='loss_test')
plt.plot(acc,'r',linewidth=2,label='acc')
plt.show()




# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')