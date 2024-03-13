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


class SpectralAttentionNet(nn.Module):
    def __init__(self, klen, hidden_dim, num_classes):
        super(SpectralAttentionNet, self).__init__()
        self.klen = klen  # Number of spectral measurements
        
        # Location-based attention components
        self.location_fc = nn.Linear(2, hidden_dim)  # Linear layer for location features
        self.location_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)

        # Content-based attention components
        self.content_fc = nn.Linear(klen, hidden_dim)  # Linear layer for spectral features
        self.content_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2)

        # Final classifier components
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)  # Combine both attentions
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, from_loader):
        # x is expected to be of shape (batch_size, 3 + klen)
        loc_features = from_loader['coords'][:,1:].float()  # Extract location features (x, y)
        spectral_features = from_loader['data'].float()  # Extract spectral features


        # Location attention
        loc_embedded = F.relu(self.location_fc(loc_features)).unsqueeze(1)  # Add sequence length dimension
        loc_attention, _ = self.location_attention(loc_embedded, loc_embedded, loc_embedded)

        # Content attention
        spectral_embedded = F.relu(self.content_fc(spectral_features)).unsqueeze(1)  # Add sequence length dimension
        content_attention, _ = self.content_attention(spectral_embedded, spectral_embedded, spectral_embedded)

        # Combine the outputs from the location and content attention mechanisms
        loc_attention = loc_attention.squeeze(1)  # Remove the sequence length dimension
        content_attention = content_attention.squeeze(1)
        combined_features = torch.cat([loc_attention, content_attention], dim=1)

        # Pass the combined features through the classifier to get the final output
        out = F.relu(self.fc1(combined_features))
        out = self.fc2(out)
        
        return out



def load(config):

    # n_channels = 1
    klen = config.input_size
    
    model = SpectralAttentionNet(klen,config.dim_hidden,2)
    
    return model, "basic_attention"


