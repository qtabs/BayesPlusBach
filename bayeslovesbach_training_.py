# -*- coding: utf-8 -*-
"""BayesLovesBach_Training_.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uESdPIPqp2T1ijCusIwICMp7XLkod26m

Info from there: https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

input_size = 108
sequence_length = 1000
num_layers = 2
hidden_size = 256

n_features = 1

learning_rate = 0.001
num_epochs = 5

num_classes = 108
batch_size = 128


class SimpleRNN(nn.Module):  # apparently the model expects us to put (batch_size, sequence_length, input_size)
    def __init__(self, input_size, num_layers, hidden_size, sequence_length, num_classes):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)
        # self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)  # x.size[0]

        out, _ = self.rnn(x, h0)
#         print(out.shape)
        out = out.reshape(out.shape[0], -1)
        # out = self.fc1(out)
        return out


class SimpleGRU(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, sequence_length=sequence_length):
        super(SimpleGRU, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out,_ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        # out = self.fc1(out)
        return out


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sequence_length=sequence_length, num_classes=num_classes):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.lstm(x,(h0, c0))
        out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleRNN(input_size, num_layers, hidden_size, sequence_length, num_classes).to(device=device)

# model = SimpleGRU().to(device=device)

# model = SimpleLSTM().to(device=device)

#Array structure for songs
#There will be two arrays of similar shape (108x1000), one for the ground truth notes and the other for the observations (that will include noise)

import numpy as np

data_sim=np.zeros((108,1000), dtype = float)

for n in range(len(data_sim)) :
    arr = np.random.normal(1, 0.5, 1000)
    for i in arr :
        if i<0.5 :
            arr[np.where(arr==i)] = 0
        else :
            arr[np.where(arr==i)] = 1
        data_sim[n] = arr

noise=np.zeros((108,1000), dtype=float)

for n in range(len(data_sim)) :
    arr = np.random.normal(0.5,0.1,1000)
    for i in arr :
      if i<0 :
        print(i)

      noise[n]=arr

noisy = data_sim + noise

min = np.min(noisy)
max = np.max(noisy)

data_sim_norm = (noisy - min) / (max - min)

dataset = torch.tensor(data_sim, dtype=torch.float32), torch.tensor(noisy, dtype=torch.float32)

train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(dataset=zeros_test, batch_size=batch_size, shuffle=False)

loss_criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

current_loss = 0
for epoch in range(num_epochs):
    for data, target in train_dataloader:
        data = data.to(device=device)
        print(data.shape)
        target = target.to(device=device)
        print(target.shape)

        score = model(target)
        loss = loss_criterion(score, target)
        current_loss = loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    print(f"At epoch: {epoch}, loss: {current_loss}")

def check_accuracy(dlr,model):

    total_correct = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in dlr:
            x = x.to(device=device)
            y = y.to(device=device)

            score = model(x)
            _,predictions = score.max(1)

            total_correct += (y==predictions).sum()
            total_samples += predictions.size(0)

    model.train()
    print(f"total samples: {total_samples} total_correct: {total_correct} accuracy : {float(total_correct/total_samples)* 100}")


check_accuracy(train_dataloader, model)
check_accuracy(test_dataloader, model)