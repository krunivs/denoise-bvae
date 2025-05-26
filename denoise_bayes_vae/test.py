# denoise_bayes_vae/test.py
# -*- encoding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
import torchaudio

from denoise_bayes_vae.bayes_vae import BayesianVAE


def load_model(model:BayesianVAE, model_file:str, device:torch.device):
    """
    load Bayesian VAE model trained
    :param model: (BayesianVAE) model
    :param model_file: (str) model file path
    :param device: (torch.device)
    :return:
    """
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model


def denoise(model: BayesianVAE,
            input_path: str,
            device: torch.device,
            input_dim: int = 16000,
            target_sr: int = 16000):
    """
    Denoise a noisy input wav file using trained model
    """
    model.eval()
    total_start_time = time.time()

    # Load waveform
    waveform, input_sr = torchaudio.load(input_path)  # shape: [1, L] or [2, L]

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if input_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Flatten to [L]
    waveform = waveform.squeeze(0)

    # Normalize input waveform (avoid divide-by-zero)
    waveform = waveform / (waveform.abs().max() + 1e-9)

    # Slice into input_dim-length chunks (e.g., 16000)
    segments = []
    total_samples = waveform.shape[0]

    for start in range(0, total_samples, input_dim):
        end = start + input_dim
        chunk = waveform[start:end]

        if chunk.shape[0] < input_dim:
            chunk = F.pad(chunk, (0, input_dim - chunk.shape[0]))

        segments.append(chunk)

    # Inference for each chunk
    denoised = []

    with torch.no_grad():
        for chunk in segments:
            input_tensor = chunk.to(device).unsqueeze(0)  # [1, 16000]
            recon, _, _ = model(input_tensor)
            recon = recon.squeeze(0).cpu()
            recon = recon / (recon.abs().max() + 1e-9)  # Output normalization
            denoised.append(recon)

    # Concatenate all denoised chunks
    denoised_waveform = torch.cat(denoised, dim=0)

    # Resample back to original SR if needed
    if input_sr != target_sr:
        resample_back = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=input_sr)
        denoised_waveform = resample_back(denoised_waveform.unsqueeze(0)).squeeze(0)

    total_time = time.time() - total_start_time
    print(f'Total inference time: {total_time:.2f} seconds')

    return denoised_waveform.numpy(), input_sr