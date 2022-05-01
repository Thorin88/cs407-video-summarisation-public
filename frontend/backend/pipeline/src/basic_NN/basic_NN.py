import torch as torch
from torch import nn

# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class BasicRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()

        # Define dimension of hidden layers
        self.hidden_dim = hidden_dim
        #set number of hidden layers
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer of given input size, and dimension/ number of hidden layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    #define a forwards pass through the network
    def forward(self, x):
        #only care about second dimension in shape
        _, n, _ = x.shape

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(n)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        #sigmoid activation function
        sig = nn.Sigmoid()
        out = sig(out)

        #return outputs of final layer and hidden layer
        return out, hidden

    #forwards pass through the network returning the ourput as a numpy array
    #instead of a tensor
    def predict(self, seq):
        seq_len = seq.shape[1]
        output = self(seq)
        # Reshape to an (N,) shaped output
        preds = output[0].view((output[0].shape[0],)).cpu().numpy()


        return preds

    # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden
