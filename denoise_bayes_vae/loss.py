# denoise_bayes_vae/loss.py
# -*- encoding: utf-8 -*-
from typing import Tuple, List

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torchaudio.transforms import MFCC

from utils.logger import Logger

logger = Logger().get_logger()


def mfcc_distance(x: Tensor, y: Tensor, sample_rate: int = 16000, n_mfcc: int = 13) -> Tensor:
    """
    Compute perceptual loss as L1 distance between MFCCs of two waveforms.

    Args:
        x (Tensor): Predicted waveform, shape (B, T)
        y (Tensor): Ground truth waveform, shape (B, T)
        sample_rate (int): Sampling rate in Hz (default: 16000)
        n_mfcc (int): Number of MFCC coefficients to compute (default: 13)

    Returns:
        Tensor: Mean absolute error between MFCCs
    """
    device = x.device

    mfcc_extractor = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 512, "hop_length": 128, "n_mels": 40},
    ).to(device)

    x_mfcc = mfcc_extractor(x)
    y_mfcc = mfcc_extractor(y)

    return F.l1_loss(x_mfcc, y_mfcc)


def log_stft_loss(
        x: Tensor,
        y: Tensor,
        n_fft: int = 512,
        hop_length: int = 128) -> Tensor:
    """
    Compute log-STFT loss between reconstructed and target waveforms.
    This captures perceptual differences in spectral content.

    Args:
        x (Tensor): Reconstructed waveform (batch, time)
        y (Tensor): Ground-truth clean waveform (batch, time)
        n_fft (int): FFT window size used in STFT (e.g., 512 for ~32ms at 16kHz)
        hop_length (int): Hop size between frames (e.g., 128 for ~8ms at 16kHz)

    Returns:
        Tensor: Mean absolute error between log-magnitude STFTs
    """
    window = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
    logX = torch.log(torch.abs(X) + 1e-7)
    logY = torch.log(torch.abs(Y) + 1e-7)

    if torch.isnan(logX).any() or torch.isnan(logY).any():
        logger.warning("[STFT] NaN detected in STFT computation")
        return torch.tensor(0.0, device=x.device)

    return F.l1_loss(logX, logY)


def multi_stft_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        fft_sizes: List[int] = None,
        hop_ratio: float = 0.2) -> torch.Tensor:
    """
    Multi-resolution STFT loss using several FFT sizes.

    Args:
        x (Tensor): predicted waveform
        y (Tensor): ground truth waveform
        fft_sizes (List[int], optional): List of FFT window sizes.
        hop_ratio (float): ratio of hop size to FFT size

    Returns:
        Tensor: average log-STFT loss across resolutions
    """
    if fft_sizes is None:
        fft_sizes = [128, 256, 512, 1024, 2048]

    loss = 0.0
    for n_fft in fft_sizes:
        hop_length = int(n_fft * hop_ratio)
        loss += log_stft_loss(x, y, n_fft=n_fft, hop_length=hop_length)

    return loss / len(fft_sizes)

def mel_stft_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> torch.Tensor:
    """
    Compute the Mel-STFT (Mel-spectrogram) based spectral loss between two waveforms.
    두 파형(음성 신호) 간의 Mel-spectrogram 기반 스펙트럼 손실(L1)을 계산합니다.

    Args:
        x (torch.Tensor): 복원(reconstructed) 파형 텐서, shape [B, T] (batch, time)
        y (torch.Tensor): 정답(target) 파형 텐서, shape [B, T]
        sample_rate (int): 오디오 샘플링 레이트 (기본값 16000)
        n_fft (int): FFT 윈도우 크기 (기본값 1024)
        hop_length (int): FFT hop length (기본값 256)
        n_mels (int): Mel filter 개수 (기본값 80)

    Returns:
        torch.Tensor: batch별 평균 Mel-STFT loss (L1 distance of log-mel)
    Example:
        loss = mel_stft_loss(denoised, clean)
    """
    device = x.device
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    ).to(device)

    x_mel = mel(x)
    y_mel = mel(y)

    # dB로 변환 (log scale)
    x_db = torch.log(x_mel + 1e-7)
    y_db = torch.log(y_mel + 1e-7)

    return F.l1_loss(x_db, y_db)


def adjust_kl_z_scale(
        kl_z_value: float,
        epoch: int,
        target_kl: float = 30.0,
        warmup_epochs: int = 20,
        max_scale: float = 0.3,
        min_scale: float = 1e-3,
        method: str = 'linear',
        kl_z_scale: float = None) -> float:
    """
    KL annealing(warm-up)과 loss 기반 adaptive scaling을 통합.
    초기엔 warm-up, 이후엔 adaptive scaling.

    Args:
        kl_z_value (float): 현재 KL 값
        epoch (int): 현재 epoch
        target_kl (float): KL 타겟
        warmup_epochs (int): warm-up 기간
        max_scale (float): KL 스케일 upper bound
        min_scale (float): KL 스케일 lower bound
        method (str): 'linear' or 'sigmoid'
        kl_z_scale (float or None): 현재 KL scale (후반부에 필요)
    Returns:
        float: 조정된 KL scale
    """
    import math
    # 1) Warm-up (schedule 기반)
    if epoch < warmup_epochs:
        if method == 'sigmoid':
            center = warmup_epochs // 2
            steepness = 5.0 / warmup_epochs
            scale = max_scale / (1 + math.exp(-steepness * (epoch - center)))
        else:  # linear
            scale = min(max_scale, (epoch + 1) / warmup_epochs * max_scale)

        return max(min_scale, scale)
    else:
        # 2) Adaptive scaling (target 기반)
        if kl_z_scale is None:
            kl_z_scale = max_scale
        if kl_z_value > target_kl * 2.0:
            scale = max(kl_z_scale * 0.5, min_scale)
        elif kl_z_value < target_kl * 0.5:
            scale = min(kl_z_scale * 1.2, max_scale)
        else:
            scale = kl_z_scale

        return scale


def adjust_stft_scale(
        stft_value: float,
        stft_scale: float,
        target_stft: float = 4.0,
        min_scale: float = 0.05,
        max_scale: float = 0.30) -> float:
    """
    Dynamically adjusts the STFT loss scale to balance perceptual fidelity during training.

    This function monitors the current STFT loss (`stft_value`) and adaptively increases or
    decreases the corresponding STFT loss weight (`stft_scale`) based on its deviation from
    the target value. If the STFT loss is too low, the scale is increased to emphasize
    frequency-domain accuracy. If too high, the scale is reduced to prevent overfitting
    to spectral details.

    Args:
        stft_value (float): Current STFT loss value for the batch or epoch.
        stft_scale (float): Current scaling weight applied to the STFT loss term.
        target_stft (float, optional): Desired STFT loss target. Default is 4.0.
        min_scale (float, optional): Minimum allowed scale value. Default is 0.05.
        max_scale (float, optional): Maximum allowed scale value. Default is 0.30.

    Returns:
        float: Updated STFT loss scaling factor.
    """
    if stft_value > target_stft * 2.0:
        stft_scale = max(min_scale, stft_scale * 0.9)

    elif stft_value < target_stft * 0.5:
        stft_scale = min(max_scale, stft_scale * 1.10)

    logger.debug(f"[STFT scale adjust] stft_value={stft_value:.4f}, new_scale={stft_scale:.5f}")
    return stft_scale


def adjust_kl_bnn_scale(
        kl_bnn_value: float,
        kl_bnn_scale: float,
        target_kl_bnn: float = 1.0) -> float:
    """
    Adjust KL loss scale for Bayesian neural network weights.

    Args:
        kl_bnn_value (float): Current KL BNN loss value
        kl_bnn_scale (float): Current scaling factor
        target_kl_bnn (float): Desired target KL BNN loss value

    Returns:
        float: Updated kl_bnn_scale.
    """
    if kl_bnn_value > target_kl_bnn * 2.0:
        kl_bnn_scale = max(kl_bnn_scale * 0.8, 1e-6)

    elif kl_bnn_value < target_kl_bnn * 0.5:
        kl_bnn_scale = min(kl_bnn_scale * 1.2, 1e-2)

    return kl_bnn_scale


def adjust_perceptual_scale(
    perceptual_value: float,
    perceptual_scale: float,
    target_perceptual: float = 1.0,
    min_scale: float = 0.05,
    max_scale: float = 0.30) -> float:
    """
    Adjust MFCC-based perceptual loss scale to control reconstruction fidelity.

    Args:
        perceptual_value (float): Current perceptual loss value.
        perceptual_scale (float): Current perceptual loss scaling factor.
        target_perceptual (float): Target MFCC loss value.
        min_scale (float): Lower bound for scale.
        max_scale (float): Upper bound for scale.

    Returns:
        float: Updated perceptual scale.
    """
    if perceptual_value > target_perceptual * 2.0:
        perceptual_scale = max(min_scale, perceptual_scale * 0.9)
    elif perceptual_value < target_perceptual * 0.5:
        perceptual_scale = min(max_scale, perceptual_scale * 1.10)

    logger.debug(f"[Perceptual scale adjust] value={perceptual_value:.4f}, "
                 f"new_scale={perceptual_scale:.5f}")
    return perceptual_scale


def elbo_loss(
        recon_x: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        kl_z_scale: float = 1.0,
        kl_bnn_loss: Tensor = None,
        kl_bnn_scale: float = 1e-3,
        stft_scale: float = 0.1,
        perceptual_scale: float = 0.3,
        epoch: int = 1,
        auto_adjust_kl: bool = True,
        auto_adjust_stft: bool = True,
        auto_adjust_kl_bnn: bool = True,
        auto_adjust_perceptual: bool = True,
        target_kl: float = 30.0,
        target_stft: float = 2.5,
        target_kl_bnn: float = 1.0,
        target_perceptual: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float]:
    """
        Compute the Evidence Lower Bound (ELBO) loss for a Bayesian VAE model with optional perceptual loss.

        This function combines multiple loss components:
        - Mean Squared Error (MSE) in the time domain
        - Multi-resolution STFT loss in the frequency domain
        - MFCC-based perceptual loss
        - KL divergence for the latent variable z
        - KL divergence for Bayesian weights (BNN)

        Each loss component can be individually scaled and some have auto-adjusting mechanisms based on target values.

        Args:
            recon_x (Tensor): Reconstructed waveform of shape (B, T)
            x (Tensor): Ground-truth clean waveform of shape (B, T)
            mu (Tensor): Latent mean from encoder
            logvar (Tensor): Latent log-variance from encoder
            kl_z_scale (float): Weight for KL divergence of latent variable
            kl_bnn_loss (Tensor): Precomputed KL divergence for Bayesian layers (if any)
            kl_bnn_scale (float): Weight for BNN KL divergence term
            stft_scale (float): Weight for STFT loss term
            perceptual_scale (float): Weight for MFCC-based perceptual loss
            epoch (int): Current training epoch (for annealing purposes)
            auto_adjust_kl (bool): Whether to adaptively adjust KL z weight
            auto_adjust_stft (bool): Whether to adaptively adjust STFT weight
            auto_adjust_kl_bnn (bool): Whether to adaptively adjust KL BNN weight
            auto_adjust_perceptual (bool): Whether to adaptively adjust perceptual weight
            target_kl (float): Target KL z loss value
            target_stft (float): Target STFT loss value
            target_kl_bnn (float): Target KL BNN loss value
            target_perceptual (float): Target Perceptual loss value

        Returns:
            Tuple:
                total_loss (Tensor): Combined loss
                mse_loss (Tensor): MSE loss
                stft_loss (Tensor): STFT loss
                kl_z_loss (Tensor): Latent KL divergence
                perceptual_loss (Tensor): MFCC-based perceptual loss
                kl_z_scale (float): Final KL z scale used
                stft_scale (float): Final STFT scale used
                kl_bnn_scale (float): Final BNN KL scale used
                perceptual_scale(float): Final MFCC-based perceptual scale used
    """
    if kl_bnn_loss is None:
        kl_bnn_loss = torch.tensor(0.0, device=x.device)

    mse_loss = F.mse_loss(recon_x, x, reduction="mean")
    stft_loss = 0.5 * multi_stft_loss(recon_x, x) + 0.5 * mel_stft_loss(recon_x, x)
    perceptual_loss = mfcc_distance(recon_x, x)  # MFCC-based perceptual loss

    kl_z_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    kl_z_loss = torch.clamp(kl_z_loss, min=1.0)

    kl_z_value = kl_z_loss.detach().item()
    stft_value = stft_loss.detach().item()
    kl_bnn_value = kl_bnn_loss.detach().item()

    if auto_adjust_kl:
        kl_z_scale = adjust_kl_z_scale(
            kl_z_value=kl_z_loss.item(),
            epoch=epoch,
            kl_z_scale=kl_z_scale)
    if auto_adjust_stft:
        stft_scale = adjust_stft_scale(stft_value, stft_scale, target_stft)
    if auto_adjust_kl_bnn:
        kl_bnn_scale = adjust_kl_bnn_scale(kl_bnn_value, kl_bnn_scale, target_kl_bnn)
    if auto_adjust_perceptual:
        perceptual_scale = adjust_perceptual_scale(perceptual_loss.item(), perceptual_scale, target_perceptual)

    total_loss = (
        mse_loss +
        (stft_scale * stft_loss) +
        (perceptual_scale * perceptual_loss) +
        (kl_z_scale * kl_z_loss) +
        (kl_bnn_scale * kl_bnn_loss)
    )

    mu_penalty = torch.mean(torch.abs(mu))
    logvar_penalty = torch.mean(torch.abs(logvar))
    total_loss += 5e-4 * (mu_penalty + logvar_penalty)

    logger.debug(
        f"[loss] mse={mse_loss:.5f}, stft={stft_value:.5f}, perceptual={perceptual_loss:.5f}, "
        f"kl_z={kl_z_value:.5f}, kl_z_scale={kl_z_scale:.5f}, "
        f"stft_scale={stft_scale:.5f}, kl_bnn={kl_bnn_value:.5f} (scale={kl_bnn_scale:.5f}), "
        f"total={total_loss:.5f}"
    )

    return total_loss, mse_loss, stft_loss, kl_z_loss, perceptual_loss, kl_z_scale, stft_scale, kl_bnn_scale, perceptual_scale
