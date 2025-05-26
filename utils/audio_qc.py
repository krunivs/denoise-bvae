import numpy as np
import torch
from pesq import pesq  # pip install pesq
from pystoi.stoi import stoi
from torchmetrics.functional.audio import signal_noise_ratio

from utils.exception import get_exception_traceback
from utils.logger import Logger


logger = Logger().get_logger()

def is_valid_waveform(waveform: torch.Tensor) -> bool:
    """
    check whether there is too much silence to calculate STOI/PESQ calculations
    :param waveform: (Tensor) waveform
    :return: (bool) True - valid, False - Invalid
    """
    return waveform.abs().max() > 1e-3 and waveform.std() > 1e-4

def compute_metrics(clean_waveform, enhanced_waveform, sr=16000):
    """
    Compute STOI, PESQ, and SNR between clean and enhanced waveforms.

    Args:
        clean_waveform (Tensor): [T]
        enhanced_waveform (Tensor): [T]
        sr (int): Sampling rate

    Returns:
        tuple: (stoi_score: float, pesq_score: float, snr: float)
    """
    clean_np = clean_waveform.detach().cpu().numpy()
    enhanced_np = enhanced_waveform.detach().cpu().numpy()

    try:
        def is_valid(x):
            return (
                x.size >= 320 and
                np.isfinite(x).all() and
                np.max(np.abs(x)) > 1e-3 and
                np.std(x) > 1e-4
            )

        if clean_np.shape[0] < 3200:
            raise ValueError("Skipped STOI/PESQ due to short input")

        if not is_valid(clean_np) or not is_valid(enhanced_np):
            raise ValueError("Invalid input: too short, too flat, or contains NaN/inf.")

        if not (is_valid_waveform(clean_waveform) and is_valid_waveform(enhanced_waveform)):
            raise ValueError("Waveform is too silent or flat")

        # Compute metrics
        snr = signal_noise_ratio(enhanced_waveform.unsqueeze(0), clean_waveform.unsqueeze(0)).item()
        stoi_score = stoi(clean_np, enhanced_np, sr, extended=False)
        pesq_score = pesq(sr, clean_np, enhanced_np, 'wb')

    except Exception as exc:
        logger.debug(
            f"[MetricError] Failed to compute PESQ/STOI:\n{get_exception_traceback(exc)}"
            f"[Input Stats] clean: std={np.std(clean_np):.5f}, max={np.max(clean_np):.3f}; "
            f"recon: std={np.std(enhanced_np):.5f}, max={np.max(enhanced_np):.3f}"
        )
        snr = 0.0
        stoi_score = 0.0
        pesq_score = 1.0

    return float(stoi_score), float(pesq_score), float(snr)
