# -*- encoding: utf-8 -*-
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from utils.logger import Logger

logger = Logger().get_logger()


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


def adjust_kl_z_scale(
        kl_z_value: float,
        kl_z_scale: float,
        epoch: int,
        target_kl: float = 30.0,
        warmup_epochs: int = 10,
        max_scale: float = 0.3,
        min_scale: float = 1e-3) -> float:
    """
    Adjust KL divergence scaling factor to prevent posterior collapse or explosion.

    Args:
        kl_z_value (float): Current KL divergence value
        kl_z_scale (float): Current KL weight
        epoch (int): Current epoch
        target_kl (float): Desired average KL divergence value
        warmup_epochs (int): Epochs to gradually increase KL scale
        max_scale (float): Maximum allowed KL scale
        min_scale (float): Minimum allowed KL scale

    Returns:
        float: Adjusted KL z scale
    """
    updated_scale = kl_z_scale

    # Warm-up: linearly increase KL scale until max_scale
    if epoch < warmup_epochs:
        updated_scale = min(max_scale, (epoch + 1) / warmup_epochs * max_scale)
    else:
        # Overshoot: KL too large → scale down
        if kl_z_value > target_kl * 2.0:
            updated_scale = max(kl_z_scale * 0.5, min_scale)

        # Undershoot: KL too small → scale up
        elif kl_z_value < target_kl * 0.5:
            updated_scale = min(kl_z_scale * 1.2, max_scale)

    logger.debug(
        f"[anneal] epoch={epoch}, raw_kl={kl_z_value:.4f}, "
        f"updated_kl_z_scale={updated_scale:.5f}"
    )

    return updated_scale


def adjust_stft_scale(
        stft_value: float,
        stft_scale: float,
        target_stft: float = 4.0) -> float:
    """
    Dynamically adjust STFT loss scale to balance perceptual fidelity.

    Args:
        stft_value (float): Current STFT loss value.
        stft_scale (float): Current STFT weight.
        target_stft (float): Reference STFT value.

    Returns:
        float: Updated STFT weight.
    """
    if stft_value > target_stft * 2.0:
        stft_scale = max(0.05, stft_scale * 0.9)

    elif stft_value < target_stft * 0.5:
        stft_scale = min(0.10, stft_scale * 1.05)  # ← increase upper bound from 0.6 to 0.10

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


def elbo_loss(
        recon_x: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        kl_z_scale: float = 1.0,
        kl_bnn_loss: Tensor = None,
        kl_bnn_scale: float = 1e-3,
        stft_scale: float = 0.1,
        epoch: int = 1,
        auto_adjust_kl: bool = True,
        auto_adjust_stft: bool = True,
        auto_adjust_kl_bnn: bool = True,
        target_kl: float = 30.0,
        target_stft: float = 2.5,
        target_kl_bnn: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor, float, float, float]:
    """
    Compute ELBO loss with multi-resolution STFT and adaptive weighting.

    Args:
        recon_x (Tensor): Reconstructed output.
        x (Tensor): Target clean waveform.
        mu (Tensor): Latent mean from encoder.
        logvar (Tensor): Latent log-variance from encoder.
        kl_z_scale (float): Weight for latent KL term.
        kl_bnn_loss (Tensor): KL loss for Bayesian weights.
        kl_bnn_scale (float): Weight for BNN KL term.
        stft_scale (float): Weight for STFT loss.
        epoch (int): Current epoch number.
        auto_adjust_kl (bool): Enable adaptive KL scaling.
        auto_adjust_stft (bool): Enable adaptive STFT scaling.
        auto_adjust_kl_bnn (bool): Enable adaptive BNN KL scaling.
        target_kl (float): Target value for KL divergence.
        target_stft (float): Target value for STFT loss.
        target_kl_bnn (float): Target value for BNN KL.

    Returns:
        Tuple of total loss, MSE, STFT, KL_z, updated KL scale, STFT scale, BNN KL scale
    """
    if kl_bnn_loss is None:
        kl_bnn_loss = torch.tensor(0.0, device=x.device)

    # Time-domain MSE loss
    mse_loss = F.mse_loss(recon_x, x, reduction="mean")

    # Multi-resolution STFT loss
    stft_loss = multi_stft_loss(recon_x, x)

    # Latent KL divergence
    kl_z_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    kl_z_loss = torch.clamp(kl_z_loss, min=1.0)  # free bits

    # KL scaling update
    kl_z_value = kl_z_loss.detach().item()
    stft_value = stft_loss.detach().item()
    kl_bnn_value = kl_bnn_loss.detach().item()

    if auto_adjust_kl:
        kl_z_scale = min(adjust_kl_z_scale(kl_z_value, kl_z_scale, epoch, target_kl), 0.3)
        if epoch <= 5:
            kl_z_scale = min(kl_z_scale, 10e-3)

    if auto_adjust_stft:
        stft_scale = adjust_stft_scale(stft_value, stft_scale, target_stft)

    if auto_adjust_kl_bnn:
        kl_bnn_scale = adjust_kl_bnn_scale(kl_bnn_value, kl_bnn_scale, target_kl_bnn)

    # Total ELBO loss
    total_loss = mse_loss + (stft_scale * stft_loss) + (kl_z_scale * kl_z_loss) + (kl_bnn_scale * kl_bnn_loss)

    # Regularization to avoid collapse
    mu_penalty = torch.mean(torch.abs(mu))
    logvar_penalty = torch.mean(torch.abs(logvar))
    total_loss += 5e-4 * (mu_penalty + logvar_penalty)

    logger.debug(
        f"[loss] mse={mse_loss:.5f}, "
        f"stft={stft_value:.5f}, "
        f"kl_z={kl_z_value:.5f}, "
        f"kl_z_scale={kl_z_scale:.5f}, "
        f"stft_scale={stft_scale:.5f}, "
        f"kl_bnn_loss={kl_bnn_value:.5f} (scale={kl_bnn_scale:.5f}), "
        f"total={total_loss:.5f}"
    )

    return total_loss, mse_loss, stft_loss, kl_z_loss, kl_z_scale, stft_scale, kl_bnn_scale
