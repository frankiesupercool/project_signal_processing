import torch
import torch.nn as nn

from video_encoding.models.swish import Swish

"""Based on Lipreading using temporal convolutional networks (https://arxiv.org/pdf/2001.08702). With
implementation from:

https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
"""

def _average_batch( x, lengths):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            Swish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, T = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
