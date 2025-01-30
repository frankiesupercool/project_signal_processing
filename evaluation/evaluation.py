import numpy as np
import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility, \
    SignalDistortionRatio

from dataset_lightning.lightning_datamodule import DataModule
from utils.device_utils import get_device

def evaluate_with_loading(model,best_model_path, data_module:DataModule):
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    evaluate(model, data_module)

def evaluate(model, data_module: DataModule):

    pesq_scores = []
    stoi_scores = []
    sdr_scores = []

    pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
    stoi = ShortTimeObjectiveIntelligibility(16000, False)
    sdr = SignalDistortionRatio()

    model.eval()

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            clean_audio = batch['clean_speech']
            encoded_audio = batch['encoded_speech']
            encoded_video = batch['encoded_video']

            encoded_audio.to(get_device())
            encoded_video.to(get_device())

            result = model.forward(encoded_audio, encoded_video)

            for i in range(clean_audio.size(0)):
                clean_sample = clean_audio[i].cpu().numpy()
                denoised_sample = result[i].cpu().numpy()
                # Ensure the signal is within appropriate range and the same length
                min_len = min(len(clean_sample), len(denoised_sample))
                clean_sample = clean_sample[:min_len]
                denoised_sample = denoised_sample[:min_len]
                # PESQ
                try:
                    pesq_score = pesq(clean_sample, denoised_sample)
                    pesq_scores.append(pesq_score)
                except Exception as e:
                    print(f"PESQ evaluation error: {e}")
                    pesq_scores.append(None)

                # STOI
                stoi_score = stoi(clean_sample, denoised_sample)
                stoi_scores.append(stoi_score)

                # SDR
                sdr_score = sdr(clean_sample, denoised_sample)
                sdr_scores.append(sdr_score)

                # Filter None values from PESQ results
            pesq_scores = [score for score in pesq_scores if score is not None]

            # Calculate average scores
            avg_pesq = np.mean(pesq_scores) if pesq_scores else float('nan')
            avg_stoi = np.mean(stoi_scores) if stoi_scores else float('nan')
            avg_sdr = np.mean(sdr_scores) if sdr_scores else float('nan')

            return avg_pesq, avg_stoi, avg_sdr


