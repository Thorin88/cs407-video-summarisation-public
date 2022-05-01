#pytorch imports
import torch
import torch.nn as nn
#class contianing the auto encoder module
class AE2D(nn.Module):
    def __init__(self, **kwargs):
        #initialise Neural net
        super().__init__()
        #initial input layer
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        #encoder hidden layer gradually decreasing size
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=512, out_features=128
        )
        #bottleneck: TAKE FEATUERS FROM HERE
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=2
        )
        #decoder hidden layer gradually increasing size
        self.decoder_hidden_layer = nn.Linear(
            in_features=2, out_features=128
        )
        #decoder hidden layer gradually increasing size
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=128, out_features=512
        )
        #final decoder layer where output error will be checked
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    #funciton degining a forwards pass through the network
    def forward(self, features):
        #run input features through network and assign a relu loss function
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        #FOR EACH LAYER:run previous layer features through network and assign a relu loss function
        hidden = self.encoder_hidden_layer2(activation)
        code = self.encoder_output_layer(hidden)
        code = torch.relu(code)

        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)

        hidden = self.decoder_hidden_layer2(activation)
        hidden = torch.relu(hidden)

        activation = self.decoder_output_layer(hidden)
        reconstructed = torch.relu(activation)
        #output reconstructed data
        return reconstructed
