import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import random
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt

from CAE import CAE

raw_data = np.load('../../data/simu_chromatography/train_set/raw_data.npy')
noise_data = np.load('../../data/simu_chromatography/train_set/noise_data.npy')
# white_noise = np.load('../../data/simu_chromatography/white_noise.npy')
# drift_noise = np.load('../../data/simu_chromatography/drift_noise.npy')

raw_data = torch.from_numpy(raw_data)
raw_data = raw_data.to(torch.float32)
noise_data = torch.from_numpy(noise_data)
noise_data = noise_data.to(torch.float32)

num_epochs = 10000
batch_size = 32
learning_rate = 1e-3
point_num = 8185

model = CAE()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=0.08)

sample_code = np.arange(noise_data.shape[0])
loss_list = np.zeros((300, 2))

# training
for epoch in range(num_epochs):
    samples = random.choices(sample_code, k=batch_size)

    # ===================forward=====================
    output = model(noise_data[samples, :point_num].view(-1, 1, point_num))
    loss = criterion(output, raw_data[samples, :point_num].view(-1, 1, point_num))

    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ===================log========================
    if epoch % 100 == 0:
        loss_list[int(epoch / 100), 0] = epoch
        loss_list[int(epoch / 100), 1] = loss.data
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))

np.save('loss_list.npy', loss_list)
torch.save(model.state_dict(), 'CAE_v1.pth')