import torch
import torch.nn as nn
import math
import numpy as np

from video_encoding.models.dense_tcn_model import DenseTCN
from video_encoding.models.resnet import ResNet, BasicBlock
from video_encoding.models.swish import Swish


class VideoEncoder(nn.Module):
    def __init__( self,
                  backbone_type='resnet',
                  num_classes=500,
                  relu_type='prelu',
                  densetcn_options={},
                  use_boundary=False,
                  extract_feats=True,):
        super(VideoEncoder, self).__init__()
        self.backbone_type = backbone_type
        self.use_boundary = use_boundary
        self.extract_feats = extract_feats

        #Standard options for ResNet
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        #Defining the different layers
        conv3d_layer = nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
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

        max_pool_3d_layer = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.frontend3D = nn.Sequential(conv3d_layer,
                                        batch_norm_3d_layer,
                                        frontend_relu,
                                        max_pool_3d_layer)


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

        # Initialise weights
        self._initialize_weights_randomly()


    def forward(self, x, lengths, boundaries=None):
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
        x = self.threeD_to_2D_tensor( x )
        #resnet
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))

        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)

        return x if self.extract_feats else self.tcn(x, lengths, B)


    def _initialize_weights_randomly(self):
        """
            Initialises all weights of the video encoder randomly. Overwritten when loading a model

            Args:

            Returns:

        """

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

    @staticmethod
    # -- auxiliary functions
    def threeD_to_2D_tensor(x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        return x.reshape(n_batch * s_time, n_channels, sx, sy)
