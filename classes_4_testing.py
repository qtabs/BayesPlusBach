import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleRNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden=None):
        if hidden is None:
          batch_size = x.size(0)
          hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, hidden = self.rnn(x, hidden)
        out = self.fc1(out)
        out = self.sig(out)
        return out, hidden

class SimpleGRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(SimpleGRU, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, hidden = self.gru(x, hidden)
        out = self.fc1(out)
        out = self.sig(out)
        return out, hidden

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden=None):
        if hidden is None:
          batch_size = x.size(0)
          hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
          c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, hidden = self.lstm(x, (hidden, c0))
        out = self.fc1(out)
        out = self.sig(out)
        return out, hidden


class BaselineNN(nn.Module):
    def __init__(self, input_size=12, num_units=12):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)  # value OR probability???????
        output = nn.Softmax(dim=1)(logits)
        return output
    

