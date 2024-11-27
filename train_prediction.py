import torch
import torch.nn as nn
from tqdm import tqdm
from classes_for_testing import SimpleRNN, SimpleLSTM, SimpleGRU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained models: rnn, gru and lstm:
input_size = 12
hidden_size = 256
num_layers = 2
learning_rate = 0.001
rnn = SimpleRNN(input_size, num_layers, hidden_size).to(device=device)
rnn.load_state_dict(torch.load('model_wts/SimpleRNNweights.pth'))
gru = SimpleGRU(input_size, num_layers, hidden_size).to(device=device)
gru.load_state_dict(torch.load('model_wts/GRUweights.pth'))
lstm = SimpleLSTM(input_size, num_layers, hidden_size).to(device=device)
lstm.load_state_dict(torch.load('model_wts/LSTMweights.pth'))

# Define linear readout layer
class LinearReadout(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LinearReadout, self).__init__()
        self.readout = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.readout(x)
        output = self.sigmoid(output)
        return output
    
# Instantiate
output_size = input_size
model = LinearReadout(hidden_size, output_size).to(device)

# Load data:
#TODO
# Create dummy data of dimensions (batch_size, sequence_length, input_size):
batch_size = 32
sequence_length = 1000
data = torch.rand((batch_size, sequence_length, input_size))

# Prepare the data to train for predicition: take the batches and shift the input data by one:
input_data = data[:, :-1, :]
ground_truth = data[:, 1:, :]
train_dataloader = torch.utils.data.DataLoader(list(zip(input_data, ground_truth)), batch_size=batch_size, shuffle=False)

# Train:
def train_model(linear, rnn, num_epochs):
    print(f"####Training readout over {rnn.__class__.__name__} model####")
    rnn.eval()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for input_data, ground_truth in tqdm(train_dataloader):
            input_data, ground_truth = input_data.to(device), ground_truth.to(device)

            _, hidden_states = rnn(input_data)
            output = linear(hidden_states)

            loss = criterion(output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader)}")


# Run the trainings:
num_epochs = 10
train_model(model, rnn, num_epochs)

# Generate the next note and compare with the ground truth
valid_data = torch.rand((1, sequence_length, input_size))
input_data = valid_data[:, :-1, :]
ground_truth = valid_data[:, 1:, :]
valid_dataloader = torch.utils.data.DataLoader(list(zip(input_data, ground_truth)), batch_size=64)

#Models in eval mode:
rnn.eval()
model.eval()
with torch.no_grad():
    scores = []
    for input_data, ground_truth in valid_dataloader:
        input_data, ground_truth = input_data.to(device), ground_truth.to(device)

        _, hidden_states = rnn(input_data)
        output = model(hidden_states)

        # Our prediction score will be computed by adding the
        # output values in the indices where the ground truth is 1
        prediction_score = output[ground_truth == 1].sum().item()
        scores.append(prediction_score)

final_score = sum(scores) / len(valid_dataloader)
print(f"Final score: {final_score}")
    




