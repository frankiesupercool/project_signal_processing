import torch
import torch.nn as nn

"""Based on Lipreading using temporal convolutional networks (https://arxiv.org/pdf/2001.08702). With 
implementation from:

https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
"""


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)
