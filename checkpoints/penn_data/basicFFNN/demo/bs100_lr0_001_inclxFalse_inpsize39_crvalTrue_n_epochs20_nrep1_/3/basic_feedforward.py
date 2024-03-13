import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from forge import flags

# flags.DEFINE_boolean(
#     "model_with_dict",
#     True,
#     "Makes model output predictions in dictionary instead of directly."
# )

class PerceptronNN(nn.Module):
	def __init__(self, input_size, num_labels):
		super(PerceptronNN, self).__init__()
		# Define the layers
		self.fc1 = nn.Linear(input_size, 128)  # First dense layer
		self.fc2 = nn.Linear(128, 64)         # Second dense layer
		# self.fc23 = nn.Linear(64,64)
		self.fc3 = nn.Linear(64, 64)          # Third dense layer
		self.fc4 = nn.Linear(64, num_labels)  # Output layer
		# self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		# Forward pass through the network
		x = F.relu(self.fc1(x))  # Activation function between layers
		x = F.relu(self.fc2(x))
		# x = F.relu(self.fc23(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)  # No activation, this will be included in the loss function
		# x = self.softmax(x)
		return x

class BasicFFNN(nn.Module):
	def __init__(self, input_size, num_labels):
		super(BasicFFNN, self).__init__()
		# Define the layers
		self.fc1 = nn.Linear(input_size, 64)  # First dense layer, reduced size
		self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization for the first layer
		self.dropout1 = nn.Dropout(0.5)  # Dropout with 50% probability
		
		self.fc2 = nn.Linear(64, 32)  # Reduced size for second dense layer
		self.bn2 = nn.BatchNorm1d(32)  # Batch Normalization for the second layer
		self.dropout2 = nn.Dropout(0.5)  # Dropout with 50% probability

		self.fc3 = nn.Linear(32,32)
		self.bn3 = nn.BatchNorm1d(32)

		self.fc4 = nn.Linear(32, num_labels)  # Output layer

	def forward(self, x):
		x = x['data'].float()
		# Forward pass through the network with activations, batch normalization, and dropout
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.dropout1(x)
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.dropout2(x)
		x = F.relu(self.bn3(self.fc3(x)))
		x = self.fc4(x)  # No activation here, use sigmoid or softmax outside if needed for binary classification
		return x

def load(config):

	# n_channels = 1
	n_features = config.input_size

	mlp = BasicFFNN(n_features,2)
	
	return mlp, "gpt_feedforward_halfLs"




