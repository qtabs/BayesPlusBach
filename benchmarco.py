import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes_4_testing import SimpleRNN, SimpleLSTM, SimpleGRU, BaselineNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from chunker import Chunker
from torch.utils.data import Dataset, DataLoader


class ChunkTensorDataset(Dataset):
    def __init__(self, targets_tensor, samples_tensor):
        self.targets_tensor = targets_tensor
        self.samples_tensor = samples_tensor

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, idx):
        return self.targets_tensor[idx], self.samples_tensor[idx]
    
    

def load_data(data_folder):
    # Functions to generate the sample from the CSV file
    chunk_size = 512
    batch_size = 16
    chromatic  = True
    noise = 0.5
    num_chunks = 1000

    chunker = Chunker(data_folder, chunk_size, batch_size, chromatic, noise)
    
    # Generate and concatenate chunks
    targets = []
    samples = []

    for _ in range(num_chunks):
        target, sample = chunker.create_chunck()
        targets.append(torch.tensor(target, dtype=torch.float32))
        samples.append(torch.tensor(sample, dtype=torch.float32))

    # Concatenate all targets and samples
    targets_tensor = torch.cat(targets)
    samples_tensor = torch.cat(samples)
    
    
    # Create Dataset and DataLoader
    dataset = ChunkTensorDataset(targets_tensor, samples_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
    



def train_model(model, train_dataloader, num_epochs):
    
    print(f"####Training model over {model.__class__.__name__} model####")
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for ground_truth, input_data in tqdm(train_dataloader):
            ground_truth, input_data = ground_truth.to(device), input_data.to(device)

            output = model(input_data)

            loss = criterion(output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader)}")


if __name__ == "__main__":

    # Load trained models: rnn, gru and lstm:
    input_size = 12
    hidden_size = 256
    num_layers = 2
    learning_rate = 0.001
    output_size = input_size
    batch_size = 32
    sequence_length = 1000

    # Define the models
    rnn = SimpleRNN(input_size, num_layers, hidden_size).to(device=device)
    gru = SimpleGRU(input_size, num_layers, hidden_size).to(device=device)
    lstm = SimpleLSTM(input_size, num_layers, hidden_size).to(device=device)
    simple_pred = BaselineNN(input_size=12).to(device=device)
    simple_inf = BaselineNN(input_size=12).to(device=device)


    # Load data of dimensions (batch_size, sequence_length, input_size):
    # data = torch.rand((batch_size, sequence_length, input_size)) # TODO: replace
    train_dataloader_orig = load_data('./train')
    test_dataloader_orig = load_data('./test') # TODO: adapt when available (which is not the case atm) (everything  ok?) (yes?) (answer?)

    # Store the scores
    final_scores = []

    # for model, name in zip([rnn, gru, lstm, simple_pred, simple_inf], ("rnn", "gru", "lstm", "simplenn_pred", "simplenn_inf")):
    for model, name in zip([simple_pred, simple_inf], ("simplenn_pred", "simplenn_inf")):
        # Train on prediction

        # Run the trainings:
        num_epochs = 10
        
        # TODO: find the correct way to shift the data once already in dataloader
        if model == simple_inf:
            train_dataloader = train_dataloader_orig
            test_dataloader = test_dataloader_orig
        else:
            # Prepare the data to train for prediction: take the batches and shift the input data by one:
            train_dataloader = train_dataloader_orig
            train_dataloader.dataset.samples_tensor = train_dataloader_orig.dataset.samples_tensor[:, :-1, :]
            train_dataloader.dataset.targets_tensor = train_dataloader_orig.dataset.targets_tensor[:, 1:, :]
            
            test_dataloader = test_dataloader_orig
            test_dataloader.dataset.samples_tensor = test_dataloader_orig.dataset.samples_tensor[:, :-1, :]
            test_dataloader.dataset.targets_tensor = test_dataloader_orig.dataset.targets_tensor[:, 1:, :]

        # Train the model
        train_model(model, train_dataloader, num_epochs)

        # Evaluate the models in eval mode:
        model.eval()
        with torch.no_grad():
            scores = []
            for ground_truth_test, input_test in test_dataloader:
                ground_truth_test, input_test = ground_truth_test.to(device), input_test.to(device)

                output = model(input_test)

                # Our prediction score will be computed by adding the
                # output values in the indices where the ground truth is 1
                prediction_score = (output[ground_truth_test == 1].sum() / torch.sum(ground_truth_test == 1)).item() # check dimensions here
                scores.append(prediction_score)

        final_score = sum(scores) / len(test_dataloader)
        print(f"Final score {name}: {final_score}")
        final_scores.append(final_score)


        # Finally, also train and eval the simple NN (BaselineNN) on inference task


    # On the train dataset, average the ground trutch to get the distribution
    train_dataloader = train_dataloader_orig
    test_dataloader = test_dataloader_orig
    average_prop = torch.mean(train_dataloader.dataset.targets_tensor, dim=(0,1))
    
    scores_average = []
    # Use these distributions to assess when comparing with the test ground truth (it doesnt really change anything but for coherence)
    for ground_truth_test, _ in test_dataloader:
        for datapoint in ground_truth_test:
            for element in datapoint:
                prediction_score = average_prop[element == 1].sum().item()
                scores_average.append(prediction_score)
    print(f"Score average distribution: {torch.tensor(scores_average).mean()}")
