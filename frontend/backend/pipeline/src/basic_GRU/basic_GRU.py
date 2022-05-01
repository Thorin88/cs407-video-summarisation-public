import torch as torch
from torch import nn

# https://blog.floydhub.com/gru-with-pytorch/
class BasicGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.2):
        super().__init__()

        # Defining hidden layer dimensions and number of layers
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # GRU Layers
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.relu = nn.ReLU()

    #define forward pass through network
    def forward(self, x):
        #set batch size
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.gru(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    #forwards pass through the network returning the ourput as a numpy array
    #instead of a tensor
    def predict(self, seq):
        seq_len = seq.shape[1]
        output = self(seq)
        # Reshape to an (N,) shaped output
        preds = output[0].view((output[0].shape[0],)).cpu().numpy()

        return preds

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
