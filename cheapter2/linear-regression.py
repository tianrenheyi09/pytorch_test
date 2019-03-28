import torch as t
from matplotlib import pyplot as plt
from IPython import display

device = t.device('cpu')

t.manual_seed(1000)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size,1,device=device)*5
    y = x*2+3+t.randn(batch_size,1,device=device)
    return x,y

x,y = get_fake_data(batch_size=16)

plt.scatter(x.squeeze().numpy(),y.squeeze().numpy())






w = t.randn(1,1).to(device)
b = t.zeros(1,1).to(device)

lr = 0.02
for ii in range(500):
    x,y = get_fake_data(batch_size=4)
    y_pred = x.mm(w)+b.expand_as(y)
    loss = 0.5*(y_pred-y)**2
    loss = loss.mean()

    dloss = 1
    dy_pred = dloss*(y_pred-y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    w.sub_(lr*dw)
    b.sub_(lr*db)
    if ii%50==0:
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=32)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print('w: ', w.item(), 'b: ', b.item())
"""
   y = x.mm(w) + b.expand_as(x)
RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'mat2'
"""

# 随机初始化参数
import numpy as np
w = t.rand(1, 1, requires_grad=True)
b = t.zeros(1, 1, requires_grad=True)
losses = np.zeros(500)

lr = 0.005  # 学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=32)

    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[ii] = loss.item()

    # backward：手动计算梯度
    loss.backward()

    # 更新参数
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # 梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii % 50 == 0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(), y2.numpy())  # true data
        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

    print(w.item(), b.item())
