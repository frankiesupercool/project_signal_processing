import os
import torch
import torch.nn as nn
from torch import Tensor
from video_encoding.models.video_encoder_model import VideoEncoder

"""Based on Lipreading using temporal convolutional networks (https://arxiv.org/pdf/2001.08702). With adapted
implementation from:

https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
"""


class VideoPreprocessingService(nn.Module):

    def __init__(self,
                 allow_size_mismatch: bool,
                 model_path: str,
                 use_boundary: bool,
                 relu_type: str,
                 num_classes: int,
                 backbone_type: str,
                 densetcn_options: dict
                 ):
        """
        Initializes the video encoder service.

        Args:
            allow_size_mismatch (bool):
            model_path (str): Path to the model file that will be loaded.
            use_boundary (bool):
            relu_type (str):
            num_classes (int):
            backbone_type (str):
            densetcn_options (dict):
        """
        super(VideoPreprocessingService, self).__init__()
        self.allow_size_mismatch = allow_size_mismatch
        self.model_path = model_path
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.densetcn_options = densetcn_options
        self.feature_extraction_mode = True

        # Create the model
        self.create_model(num_classes=self.num_classes,
                          densetcn_options=self.densetcn_options,
                          backbone_type=self.backbone_type,
                          relu_type=self.relu_type,
                          use_boundary=self.use_boundary,
                          extract_feats=self.feature_extraction_mode)
        # Load the model from the lrw_resnet18_dctcn_video_boundary.pth file
        self.model = self.load_model(load_path=self.model_path,
                                     model=self.model,
                                     allow_size_mismatch=self.allow_size_mismatch)

    def create_model(self,
                     num_classes: int,
                     densetcn_options: dict,
                     backbone_type: str = 'resnet',
                     relu_type: str = 'relu',
                     use_boundary: bool = False,
                     extract_feats: bool = True) -> VideoEncoder:
        """
            Creates the model using the in the paper given specifications of using a resnet and swish relu.
            It is essential that extract_feats is set to True such that the model only does the feature extraction.

            Args:
                num_classes (int): Number of classes.
                densetcn_options (dict): Options for the temporal convolutional network.
                backbone_type (str):
                relu_type (str): Type of activation function to use.
                use_boundary: (bool):
                extract_feats (bool): Whether to extract features or not. Needs to be true if this is supposed to be
                                    used as an encoder.
            Returns:
                VideoEncoder: Model instance.
        """
        # Initialise Model
        self.model = VideoEncoder(num_classes=num_classes,
                                  densetcn_options=densetcn_options,
                                  backbone_type=backbone_type,
                                  relu_type=relu_type,
                                  use_boundary=use_boundary,
                                  extract_feats=extract_feats)

    def generate_encodings(self, data: Tensor) -> Tensor:
        """
            Helper function for generating video encodings and pushing features to the device.
            Args:
                data (Tensor): Preprocessed video input.
            Returns:
                Tensor: Encoded video features.
        """
        encoded = self.extract_feats(self.model, data)
        return encoded

    @staticmethod
    def load_model(load_path: str,
                   model: VideoEncoder,
                   allow_size_mismatch=False, optimizer=None) -> VideoEncoder:
        """
            Load model from file
            If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
            If optimizer not passed, only the model weights will be loaded
            Args:
                load_path (str): Path to the model file.
                model (VideoEncoder): Model to be loaded.
                allow_size_mismatch (bool):
            Returns:
                VideoEncoder: Model with weights loaded.
        """

        # -- load dictionary
        assert os.path.isfile(load_path), "Error when loading the model, provided path not found: {}".format(load_path)
        checkpoint = torch.load(load_path, weights_only=True)
        loaded_state_dict = checkpoint['model_state_dict']

        if allow_size_mismatch:
            loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
            model_state_dict = model.state_dict()
            model_sizes = {k: v.shape for k, v in model_state_dict.items()}
            mismatched_params = []
            for k in loaded_sizes:
                if loaded_sizes[k] != model_sizes[k]:
                    mismatched_params.append(k)
            for k in mismatched_params:
                del loaded_state_dict[k]

        # -- copy loaded state into current model and, optionally, optimizer
        model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer, checkpoint['epoch_idx'], checkpoint
        return model

    @staticmethod
    def extract_feats(model: VideoEncoder, data) -> Tensor:
        """
            Method used to extract the features of the model before they are passed to the TCN. Used as an encoder.

            Args:
                model (VideoEncoder): The model whos features are extracted.
                data: Input data.

            Returns:
                Tensor: Features of the model.
        """
        assert model.extract_feats == True
        # Avoid using torch.tensor on existing tensors. Use clone().detach() if data is a tensor,
        # or torch.from_numpy() if data is a NumPy array.
        if isinstance(data, torch.Tensor):
            input_tensor = data.clone().detach().float()
        else:
            input_tensor = torch.from_numpy(data).float()

        # Add channel dimension at position 1: [batch_size, 1, frames, 96, 96]
        input_tensor = input_tensor.unsqueeze(1)

        input_tensor = input_tensor

        # Assuming all samples have the same number of frames
        lengths = [data.shape[1]] * data.shape[0]  # [frames, frames, ..., frames]
        with torch.no_grad():
            # Run the model
            output = model(input_tensor, lengths=lengths)

        return output
