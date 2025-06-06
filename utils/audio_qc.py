# utils/audio_qc.py
# -*- encoding: utf-8 -*-

import numpy as np
from pesq import pesq  # pip install pesq
from pystoi.stoi import stoi
from torchmetrics.functional.audio import signal_noise_ratio

from utils.exception import get_exception_traceback
from utils.logger import Logger


logger = Logger().get_logger()

def compute_metrics(clean_waveform, enhanced_waveform, sr=16000):
    """
    Compute average STOI, PESQ, and SNR between clean and enhanced waveforms (batched).

    Args:
        clean_waveform (Tensor): [B, T]
        enhanced_waveform (Tensor): [B, T]
        sr (int): Sampling rate

    Returns:
        tuple: (avg_stoi: float, avg_pesq: float, avg_snr: float)
    """
    clean_waveform = clean_waveform.detach()
    enhanced_waveform = enhanced_waveform.detach()
    clean_np = clean_waveform.cpu().numpy()
    enhanced_np = enhanced_waveform.cpu().numpy()

    stoi_scores = []
    pesq_scores = []
    snr_scores = []

    try:
        batch_size = clean_np.shape[0]
        for i in range(batch_size):
            c = clean_waveform[i]
            e = enhanced_waveform[i]
            c_np = clean_np[i]
            e_np = enhanced_np[i]

            # Check signal length (at least 3200 samples)
            if c_np.shape[0] < 3200 or e_np.shape[0] < 3200:
                raise ValueError(f"[waveform] Skipped due to short input (len={c_np.shape[0]})")

            # Check numerical validity
            if not np.isfinite(c_np).all():
                raise ValueError("[clean_waveform] Invalid: NaN or inf detected")
            if not np.isfinite(e_np).all():
                raise ValueError("[enhanced_waveform] Invalid: NaN or inf detected")

            # Check amplitude range and flatness
            if c.abs().max() <= 1e-3 or c.std() <= 1e-4:
                raise ValueError(f"[clean_waveform] Too silent or flat (max={c.abs().max():.5f}, std={c.std():.5f})")
            if e.abs().max() <= 1e-3 or e.std() <= 1e-4:
                raise ValueError(f"[enhanced_waveform] Too silent or flat (max={e.abs().max():.5f}, std={e.std():.5f})")

            # Compute metrics
            snr = signal_noise_ratio(e.unsqueeze(0), c.unsqueeze(0)).item()
            stoi_score = stoi(c_np, e_np, sr, extended=False)
            pesq_score = pesq(sr, c_np, e_np, 'wb')

            stoi_scores.append(stoi_score)
            pesq_scores.append(pesq_score)
            snr_scores.append(snr)

    except Exception as exc:
        logger.debug(
            f"[MetricError] Failed to compute PESQ/STOI:\n{get_exception_traceback(exc)}"
            f"[Input Stats] clean: std={np.std(clean_np):.5f}, max={np.max(clean_np):.3f}; "
            f"recon: std={np.std(enhanced_np):.5f}, max={np.max(enhanced_np):.3f}")
        return None, None, None

    # 평균 반환
    return (
        float(np.mean(stoi_scores)) if stoi_scores else None,
        float(np.mean(pesq_scores)) if pesq_scores else None,
        float(np.mean(snr_scores)) if snr_scores else None
    )
