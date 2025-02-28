import torch
import torch.nn as nn

from video_encoding.models.densetcn import DenseTemporalConvNet

"""Based on Lipreading using temporal convolutional networks (https://arxiv.org/pdf/2001.08702). With 
implementation from:

https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
"""


class DenseTCN(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                 kernel_size_set, dilation_size_set,
                 dropout, relu_type,
                 squeeze_excitation=False,
                 ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                              kernel_size_set, dilation_size_set,
                                              dropout=dropout, relu_type=relu_type,
                                              squeeze_excitation=squeeze_excitation,
                                              )
        self.tcn_output = nn.Linear(num_features, num_classes)

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self._average_batch(x, lengths, B)
        return self.tcn_output(x)

    @staticmethod
    def _average_batch(x, lengths, B):
        return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)
