import torch
import torch.nn as nn
from torch import Tensor

from video_encoding.models.dense_tcn_model import DenseTCN
from video_encoding.models.resnet import ResNet, BasicBlock
from video_encoding.models.swish import Swish

"""Based on Lipreading using temporal convolutional networks (https://arxiv.org/pdf/2001.08702). With adapted
implementation from:

https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
"""


class VideoEncoder(nn.Module):
    def __init__(self,
                 backbone_type='resnet',
                 num_classes=500,
                 relu_type='prelu',
                 densetcn_options={},
                 use_boundary=False,
                 extract_feats=True):

        """
        Constructor for the video encoder model.

        Args:
            backbone_type (str): Backbone used in the encoder. ResNet standard as given in the paper.
            num_classes (int): Number of classes in the dataset.
            relu_type (str): Type of ReLU used in the encoder. Prelu as given in the paper.
            densetcn_options (dict): Dictionary of options for the densetcn.
            use_boundary (bool): Whether to use boundary features or not.
            extract_feats (bool): Whether to extract features or not. Needs to be true if used as an encoder.
        """
        super(VideoEncoder, self).__init__()
        self.backbone_type = backbone_type
        self.use_boundary = use_boundary
        self.extract_feats = extract_feats

        #Standard options for ResNet
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        #Defining the different layers
        #Start with a 3d convolutional layer
        conv3d_layer = nn.Conv3d(1,
                                 self.frontend_nout,
                                 kernel_size=(5, 7, 7),
                                 stride=(1, 2, 2),
                                 padding=(2, 3, 3),
                                 bias=False)
        #3d Batch normalization layer
        batch_norm_3d_layer = nn.BatchNorm3d(self.frontend_nout)

        # Different types of ReLu
        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU(self.frontend_nout)
        elif relu_type == 'swish':
            frontend_relu = Swish()
        else:
            raise NotImplementedError

        #Lastly use a 3d MaxPoolLayer
        max_pool_3d_layer = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Combine all four layers
        self.frontend3D = nn.Sequential(conv3d_layer,
                                        batch_norm_3d_layer,
                                        frontend_relu,
                                        max_pool_3d_layer)

        # Dense TCN layer unused if used just to extract features
        self.tcn =  DenseTCN( block_config=densetcn_options['block_config'],
                              growth_rate_set=densetcn_options['growth_rate_set'],
                              input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                              reduced_size=densetcn_options['reduced_size'],
                              num_classes=num_classes,
                              kernel_size_set=densetcn_options['kernel_size_set'],
                              dilation_size_set=densetcn_options['dilation_size_set'],
                              dropout=densetcn_options['dropout'],
                              relu_type=relu_type,
                              squeeze_excitation=densetcn_options['squeeze_excitation'],
                            )


    def forward(self, x, lengths, boundaries=None) -> Tensor:
        """
            Standard forward pass for the model. Starts by using a CNN then the resnet and lastly the tcn. TCN is not
            used if model is used just for encoding.

            Args:
                x: input to the model
                lengths: lengths of the sequence
                boundaries: boundaries of the sequence

            Returns:
                Tensor: output of the tcn or dtcn. Not used if the model is just used as a encoder.

        """
        # x is now [batch_size, frames, 96, 96]
        # If you want an explicit channel dimension:
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]    # output should be B x C2 x Tnew x H x W
        x = self.threed_to_2d_tensor(x)
        #resnet
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))

        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)
        # For encoding self.extract_feats needs to be True. This causes the model to not run the tcn but output the
        # encodings.
        return x

    @staticmethod
    # -- auxiliary functions
    def threed_to_2d_tensor(x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        return x.reshape(n_batch * s_time, n_channels, sx, sy)
