# -*- encoding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
import torchaudio
import librosa

def load_audio_librosa_resample(path, sr=16000, duration=None, offset=None):
    """
    Load audio from a specific offset with a fixed duration.

    :param path: path to audio file
    :param sr: target sampling rate
    :param duration: duration in seconds
    :param offset: start time in seconds
    :return: waveform tensor (1, N), sampling rate
    """
    y, orig_sr = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    y = torch.tensor(y).unsqueeze(0)
    return y, sr

# slice waveform
def slice_waveform(waveform, target_length):
    """
    slice wave audio
    :param waveform:
    :param target_length:
    :return:
    """
    slices = []
    total_samples = waveform.shape[0]

    for start in range(0, total_samples, target_length):
        end = start + target_length
        slice_wave = waveform[start:end]

        if slice_wave.shape[0] < target_length:
            pad_size = target_length - slice_wave.shape[0]
            slice_wave = F.pad(slice_wave, (0, pad_size))

        slices.append(slice_wave)

    return slices

def normalize_waveform(waveform):
    """
    normalize wav form
    :param waveform:
    :return:
    """
    max_val = waveform.abs().max()

    if max_val > 0:
        waveform = waveform / max_val

    return waveform


def save_wav(waveform, filepath, sample_rate=16000):
    """
    Save waveform as .wav file
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {waveform.shape}")

    torchaudio.save(filepath, waveform, sample_rate)
    print(f'Saved denoised output to {filepath}')
