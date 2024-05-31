"""
Model definition file
"""

import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder class
    """

    def __init__(self, num_input_feat: int, emb_dim: int):
        """
        Initialization

        Args:
            num_input_feat: int, sample dimension
            emb_dim: int, dimension of the encoding
        """
        super().__init__()
        self.fc1 = nn.Linear(num_input_feat, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder pass

        Args:
            x: input data, shape (batch_size, num_input_feat)

        Return:
            x: embedding, shape (batch_size, emb_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class Decoder(nn.Module):
    """
    Decoder class
    """

    def __init__(self, num_input_feat: int, emb_dim: int):
        """
        Initialization

        Args:
            num_input_feat: int, dimension of the sample
            emb_dim: int, dimension of the encoding
        """
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_input_feat)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decoder pass

        Args:
            embedding: embeddings obtained from encoder, shape (batch_size, emb_dim)

        Returns:
            x: reconstructed value, shape (batch_size, num_input_feat)
        """
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class AutoEncoder(nn.Module):
    """
    Combine encoder and decoder
    """

    def __init__(self, num_input_feat: int, emb_dim: int):
        """
        Initialize encoder and decoder

        Args:
            num_input_feat: int, dimension of the sample
            emb_dim: int, dimension of the encoding
        """
        super().__init__()
        self.encoder = Encoder(num_input_feat, emb_dim)
        self.decoder = Decoder(num_input_feat, emb_dim)
