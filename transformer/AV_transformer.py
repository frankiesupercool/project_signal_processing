# AV_transformer.py

import pytorch_lightning as pl
from torch import nn, optim
import torch

# Define the AudioVideoTransformer Lightning Module
class AudioVideoTransformer(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super(AudioVideoTransformer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()  # Using Mean Squared Error loss

    def forward(self, encoded_audio, encoded_video):
        return self.model(encoded_audio, encoded_video)

    def training_step(self, batch, batch_idx):
        encoded_audio = batch['encoded_audio']  # [batch_size, seq_len, audio_dim=1024]
        encoded_video = batch['encoded_video']  # [batch_size, seq_len, video_dim=256]
        clean_speech = batch['clean_speech']    # [batch_size, 64000]

        # Forward pass
        predicted_clean = self(encoded_audio, encoded_video)  # [batch_size, 64000]

        # Compute loss
        loss = self.criterion(predicted_clean, clean_speech)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        # Forward pass
        predicted_clean = self(encoded_audio, encoded_video)

        # Compute loss
        loss = self.criterion(predicted_clean, clean_speech)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        encoded_audio = batch['encoded_audio']
        encoded_video = batch['encoded_video']
        clean_speech = batch['clean_speech']

        # Forward pass
        predicted_clean = self(encoded_audio, encoded_video)

        # Compute loss
        loss = self.criterion(predicted_clean, clean_speech)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Create a dummy batch to test the model
dummy_batch = {
    'encoded_audio': torch.randn(4, 61, 1024),    # [batch_size=4, seq_len=61, audio_dim=1024]
    'encoded_video': torch.randn(4, 100, 256),    # [batch_size=4, seq_len=100, video_dim=256]
    'clean_speech': torch.randn(4, 64000)         # [batch_size=4, 64000]
}

# # Instantiate the TransformerModel with correct parameters
# # Ensure that 'denoiser_decoder' is either provided or left as None
# transformer_model_instance = TransformerModel(
#     audio_dim=1024,         # From your encoded_audio
#     video_dim=256,          # From your encoded_video
#     embed_dim=768,          # As per your specification
#     nhead=8,                # As per your specification
#     num_layers=3,           # As per your specification
#     dim_feedforward=532,    # As per your specification
#     max_seq_length=1024,    # Adjust based on your sequence lengths
#     denoiser_decoder=None   # Provide the denoiser's decoder if available
# )
#
# # Instantiate the Lightning Module with the model instance
# model = AudioVideoTransformer(model=transformer_model_instance)
#
# # Forward pass with dummy data to verify
# model.eval()
# with torch.no_grad():
#     predicted_clean = model(dummy_batch['encoded_audio'], dummy_batch['encoded_video'])
#     print(predicted_clean.shape)  # Expected: [4, 64000]
