import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from forge import flags

# flags.DEFINE_integer(
#     "in_channels",
#     32,
#     "Number of output classes."
# )

class Conv2D(nn.Module):
    def __init__(self, input_channels):
        super(Conv2D, self).__init__()
        # Reduced convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        # Transition layer to get 2 output channels while keeping spatial dimensions
        self.transition = nn.Conv2d(32, 2, 1)  # 1x1 convolution
        # Optional: Batch normalization layers (comment out if not needed to reduce parameters)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        # Applying convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Transition to 2 channels
        x = self.transition(x)
        return x

    @torch.no_grad()
    def predict(self, x): # this method is required by the trainer from the Unet backbones package
        """
        Inference method. Switch model to `eval` mode, 
        call `.forward(x)` with `torch.no_grad()`
        Parameters
        ----------
        x: torch.Tensor
            4D torch tensor with shape (batch_size, channels, height, width)
        Returns
        ------- 
        prediction: torch.Tensor
            4D torch tensor with shape (batch_size, classes, height, width)
        """
        self.eval()
        x = self.forward(x)
        return x



def load(config):

    model = Conv2D(config.input_channels)

    
    return model, "Conv2D"

