import torchvision
import torch
import torch.nn as nn
from ae2d_class import AE2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


###########################INSTANTIATE OBJECTS#########################################
#  use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE2D(input_shape=1024).to(device)

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

#register hook to get middle layer
model.encoder_output_layer.register_forward_hook(get_features('feats'))


# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()


##########################################LOAD DATA##################################

# Runs in AE2D directory
# New data that is definitely 1024D with sample_rate == 1
video_feats = np.load("./../../../dsnet_feature_extraction/features/combined_features_avg.npy")

#convert numpy arrat to tensor
video_Tensors = torch.from_numpy(video_feats)
print(video_Tensors)

#create dataset of tensors
video_dataset = torch.utils.data.TensorDataset(video_Tensors)


#create train and test dataloaders
train_loader = torch.utils.data.DataLoader(
    video_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    video_dataset, batch_size=10, shuffle=False, num_workers=0
)

###################################TRAINING##############################################
#declare number of epochs
epochs = 500
#store features here
FEATS = []
# placeholder for batch features
features = {}

for epoch in range(epochs):
    #only care about the features extracted from the most recent epoch
    #FEATS = []
    loss = 0
    for batch_features in train_loader:
        batch_feat = batch_features[0]
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_feat = batch_feat.view(-1,1024).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_feat)

        #extract feature vectors from bottleneck
        #FEATS.append(features['feats'].cpu().numpy())

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_feat)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


#set model to eval mode to prevent it from updating weights
model.eval()
final_loss = 0
#only loop through data here once
for batch_features in test_loader:
    batch_feat = batch_features[0]
    # reshape mini-batch data to [N, 784] matrix
    # load it to the active device
    batch_feat = batch_feat.view(-1, 1024).to(device)

    # compute reconstructions
    outputs = model(batch_feat)

    #extract feature vectors from bottleneck
    FEATS.append(features['feats'].cpu().numpy())

    # compute training reconstruction loss
    train_loss = criterion(outputs, batch_feat)

    # add the mini-batch training loss to epoch loss
    final_loss += train_loss.item()

#calculate and print final test loss
final_loss = final_loss / len(test_loader)
print("final test loss = {:.6f}".format(final_loss))
FEATS = np.concatenate(FEATS)
print('- feats shape:', FEATS.shape)
# print("feats: " ,FEATS)
#save 2D featutues
np.save("FEATS", FEATS)
# Saved weights
torch.save(model.state_dict(), "pretrainedAE2D.pt")
