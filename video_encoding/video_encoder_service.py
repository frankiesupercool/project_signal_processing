import os
import torch

from video_encoding.models.video_encoder_model import VideoEncoder

device = torch.device("cpu")



class VideoPreprocessingService:

    def __init__(self,
                 allow_size_mismatch: bool,
                 model_path: str,
                 use_boundary: bool,
                 relu_type: str,
                 num_classes: int,
                 backbone_type: str,
                 densetcn_options
                 ):
        self.allow_size_mismatch = allow_size_mismatch
        self.model_path = model_path
        self.use_boundary = use_boundary
        self.relu_type = relu_type
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.densetcn_options = densetcn_options

        self.create_model()
        self.model = self.load_model(self.model_path, self.model, allow_size_mismatch=self.allow_size_mismatch)



    def create_model(self):

        # Define model parameters form json lrw_resnet18_dctcn_boundary.json
        backbone_type = "resnet"
        relu_type = "swish"
        use_boundary = False

        # Initialise Model
        self.model = VideoEncoder(num_classes=self.num_classes,
                           densetcn_options=self.densetcn_options,
                           backbone_type=backbone_type,
                           relu_type=relu_type,
                           use_boundary=use_boundary).to(device)

    def generate_encodings(self, data):
        encoded = self.extract_feats(self.model, data)
        # print(f"Raw encoded shape: {encoded.shape}")  # Debugging statement
        return encoded.to(device).detach().numpy()

    @staticmethod
    def load_model(load_path, model, optimizer=None, allow_size_mismatch=False):
        """
        Load model from file
        If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
        If optimizer not passed, only the model weights will be loaded
        """

        # -- load dictionary
        assert os.path.isfile(load_path), "Error when loading the model, provided path not found: {}".format(load_path)
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'), weights_only=True)
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
    def extract_feats(model, data):
        input_tensor = torch.FloatTensor(data)[None, None, :, :, :].to(device)  # Shape: [1, 1, 100, 96, 96]
        lengths = [data.shape[0]]
        output = model(input_tensor, lengths=lengths)

        # print(f"Model output shape: {output.shape}")  # Debugging statement
        return output


