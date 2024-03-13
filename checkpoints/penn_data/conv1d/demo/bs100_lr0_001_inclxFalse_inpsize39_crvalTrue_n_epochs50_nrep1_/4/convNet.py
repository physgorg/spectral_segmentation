import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from forge import flags

flags.DEFINE_integer(
    "dim_hidden",
    32,
    "Width of hidden layer."
)
flags.DEFINE_integer(
    "n_classes",
    2,
    "Number of output classes."
)
flags.DEFINE_integer(
    "kernel_size",
    3,
    "Size of 1D conv kernel."
)



class ConvNN(nn.Module):
    def __init__(self, dim_hidden, kernel_size,num_classes):
        super(ConvNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(1,dim_hidden,kernel_size)  # First convolution layer
        self.conv2 = nn.Conv1d(dim_hidden,dim_hidden,2*kernel_size) # second convolution layer

        self.fc1 = nn.Linear(dim_hidden,dim_hidden)
        self.fc2 = nn.Linear(dim_hidden,dim_hidden)
        self.fc3 = nn.Linear(dim_hidden,num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x['data']
        x = torch.unsqueeze(x,1).float() # add a single channel index

        # Forward pass through the network
        x = F.relu(self.conv1(x))  # Activation function between layers
        
        x = F.relu(self.conv2(x))

        global_max_pool = nn.MaxPool1d(kernel_size=x.size()[2])  # This takes the size of the third dimension
        x = global_max_pool(x).squeeze(2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



def load(config):

    model = ConvNN(config.dim_hidden, config.kernel_size,config.n_classes)

    
    return model, "conv1d"