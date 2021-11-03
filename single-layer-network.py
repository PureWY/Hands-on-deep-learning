import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
  # 构造一个pytorch数据迭代器
  dataset = data.TensorDataset(*data_arrays)
  return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# print(next(iter(data_iter)))

# 使用框架的预定义好的层
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)   # weight: 权重，data：真实data，用均方分布替换掉真实的data
net[0].bias.data.fill_(0)   # 设置偏差

# 使用均方误差
loss = nn.MSELoss()

# 实例化SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# run
num_epochs = 3
for epoch in range(num_epochs):
  for X, y in data_iter:
    l = loss(net(X), y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
  l = loss(net(features), labels)
  print(f'epoch {epoch + 1}, loss {l:f}')


