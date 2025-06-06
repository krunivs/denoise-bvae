# utils/report.py
# -*- encoding: utf-8 -*-

import os

from torch import Tensor


def generate_train_report(
        mu,
        logvar,
        z,
        recon,
        recon_cos_sim_mean,
        alpha,
        stoi: float,
        pesq: float,
        snr: float,
        epoch,
        kl_z_scale,
        stft_scale,
        kl_bnn_scale,
        perceptual_scale,
        mse_loss=None,
        stft_loss=None,
        kl_z_loss=None,
        kl_bnn_loss=None,
        perceptual_loss=None,
        total_loss=None,
        output_file=None):
    """
    Generate and log training statistics to monitor VAE collapse behaviors and loss dynamics.

    This function computes and records statistical indicators of model health during training of a Bayesian VAE.
    It helps identify common failure modes such as:
    - Posterior collapse (near-zero std of z or mu)
    - Prior collapse (near-zero std of logvar)
    - Decoder collapse (low output variance)
    - Decoder repetition (high cosine similarity between reconstructions)

    Additionally, it records the current values of:
    - Loss terms (MSE, STFT, KL_z, KL_BNN, total)
    - Scaling weights used for each loss component (kl_z_scale, stft_scale, kl_bnn_scale)

    Args:
        mu (Tensor): Mean vector from encoder output (shape: [batch, latent_dim])
        logvar (Tensor): Log variance from encoder output (shape: [batch, latent_dim])
        z (Tensor): Sampled latent vectors from approximate posterior (shape: [batch, latent_dim])
        recon (Tensor): Reconstructed output waveform (shape: [batch, time])
        recon_cos_sim_mean (float): Reconstructed output cosine similarity mean
        alpha (float): output weight for decoder
        stoi (float): Average Short-Time Objective Intelligibility (STOI) score.
        pesq (float): Average PESQ score.
        snr (float): Average Signal-to-Noise Ratio.
        epoch (int): Current training epoch (1-based)
        kl_z_scale (float): Scaling weight for latent KL divergence loss
        stft_scale (float): Scaling weight for STFT loss
        kl_bnn_scale (float): Scaling weight for Bayesian weight prior KL loss
        perceptual_scale (float): Scaling weight for MFCC-based perceptual loss
        mse_loss (float, optional): Current MSE loss value
        stft_loss (float, optional): Current STFT loss value
        kl_z_loss (float, optional): Current KL divergence loss for latent z
        kl_bnn_loss (float, optional): Current KL divergence loss for Bayesian weights
        perceptual_loss (float): MFCC-based perceptual loss
        total_loss (float, optional): Current total ELBO loss
        output_file (str, optional): Path to TSV file to append logging information

    TSV Output Columns:
        - epoch
        - mu_mean, mu_std
        - logvar_mean, logvar_std
        - z_mean, z_std
        - recon_mean, recon_std
        - recon_cos_sim (mean cosine similarity across reconstructions)
        - kl_z_scale, stft_scale, kl_bnn_scale
        - mse_loss, stft_loss, kl_z_loss, kl_bnn_loss, total_loss

    Returns:
        None. Results are written to the specified TSV file.
    """
    stats = {
        "epoch": epoch,
        "mu_mean": mu.detach().mean().item(),
        "mu_std": mu.detach().std().item(),
        "logvar_mean": logvar.detach().mean().item(),
        "logvar_std": logvar.detach().std().item(),
        "z_mean": z.detach().mean().item(),
        "z_std": z.detach().std().item(),
        "recon_mean": recon.detach().mean().item(),
        "recon_std": recon.detach().std().item(),
        "recon_cos_sim": recon_cos_sim_mean,
        "alpha": alpha,
        "stoi": stoi,
        "pesq": pesq,
        "snr": snr,
        "kl_z_scale": kl_z_scale,
        "stft_scale": stft_scale,
        "kl_bnn_scale": kl_bnn_scale,
        "perceptual_scale": perceptual_scale,
        "mse_loss": float(mse_loss) if mse_loss is not None else -1,
        "stft_loss": float(stft_loss) if stft_loss is not None else -1,
        "kl_z_loss": float(kl_z_loss) if kl_z_loss is not None else -1,
        "kl_bnn_loss": float(kl_bnn_loss) if kl_bnn_loss is not None else -1,
        "perceptual_loss": perceptual_loss if perceptual_loss is not None else -1,
        "total_loss": float(total_loss) if total_loss is not None else -1,
    }
    if output_file is None:
        return

    # Save summary to tsv
    write_header = not os.path.exists(output_file)

    with open(output_file, "a") as f:
        if write_header:
            header = "\t".join(stats.keys()) + "\n"
            f.write(header)
        f.write("\t".join([f"{stats[k]:.4f}" if isinstance(stats[k], float) else str(stats[k]) for k in stats]) + "\n")


def generate_validation_report(
        epoch: int,
        mu: Tensor,
        logvar: Tensor,
        z: Tensor,
        recon: Tensor,
        recon_cos_sim_mean,
        alpha,
        stoi: float,
        pesq: float,
        snr: float,
        mse_loss: float = -1.0,
        stft_loss: float = -1.0,
        stft_scale: float = -1.0,
        kl_z_loss: float = -1.0,
        kl_z_scale: float = -1.0,
        kl_bnn_loss: float = -1.0,
        kl_bnn_scale: float = -1.0,
        perceptual_loss: float = -1.0,
        perceptual_scale: float = -1.0,
        total_loss: float = -1.0,
        output_file: str = None):
    """
    Generate and log validation statistics to monitor model health and performance.

    This function records numerical indicators of latent variable behavior and reconstruction quality
    during validation. It is designed to identify signs of posterior/prior collapse and track
    loss dynamics across epochs. The format is aligned with `generate_train_report()` for consistency.

    Args:
        epoch (int): Current epoch number.
        mu (Tensor): Mean vector from encoder output (shape: [batch, latent_dim])
        logvar (Tensor): Log variance from encoder output (shape: [batch, latent_dim])
        z (Tensor): Sampled latent vectors from approximate posterior (shape: [batch, latent_dim])
        recon (Tensor): Reconstructed output waveform (shape: [batch, time])
        recon_cos_sim_mean (float): Reconstructed output cosine similarity mean
        alpha (float): output weight for decoder
        stoi (float): Average Short-Time Objective Intelligibility (STOI) score.
        pesq (float): Average PESQ score.
        snr (float): Average Signal-to-Noise Ratio.
        mse_loss (float, optional): MSE loss value (default: -1.0).
        stft_loss (float, optional): STFT loss value (default: -1.0).
        stft_scale (float, optional): STFT loss scale value (default: -1.0).
        kl_z_loss (float, optional): Latent KL divergence loss (default: -1.0).
        kl_z_scale (float, optional): Latent KL divergence loss scale value (default: -1.0).
        kl_bnn_loss (float, optional): KL loss from Bayesian layers (default: -1.0).
        kl_bnn_scale (float, optional): KL loss scale value from Bayesian layers (default: -1.0).
        perceptual_loss (float, optional): MFCC-based Perceptual loss value (default: -1.0).
        perceptual_scale (float, optional): MFCC-based Perceptual loss scale value (default: -1.0).
        total_loss (float, optional): Total ELBO loss (default: -1.0).
        output_file (str, optional): Path to the TSV output file.

    TSV Output Columns:
        - epoch
        - mu_mean, mu_std
        - logvar_mean, logvar_std
        - z_mean, z_std
        - recon_mean, recon_std
        - stoi, pesq, snr
        - mse_loss, stft_loss, kl_z_loss, kl_bnn_loss, total_loss

    Returns:
        None. Results are written to the specified TSV file.
    """
    if output_file is None:
        return

    stats = {
        "epoch": epoch,
        "mu_mean": mu.detach().mean().item(),
        "mu_std": mu.detach().std().item(),
        "logvar_mean": logvar.detach().mean().item(),
        "logvar_std": logvar.detach().std().item(),
        "z_mean": z.detach().mean().item(),
        "z_std": z.detach().std().item(),
        "recon_mean": recon.detach().mean().item(),
        "recon_std": recon.detach().std().item(),
        "recon_cos_sim": recon_cos_sim_mean,
        "alpha": alpha,
        "stoi": stoi,
        "pesq": pesq,
        "snr": snr,
        "mse_loss": mse_loss,
        "stft_loss": stft_loss,
        "stft_scale": stft_scale,
        "kl_z_loss": kl_z_loss,
        "kl_z_scale": kl_z_scale,
        "kl_bnn_loss": kl_bnn_loss,
        "kl_bnn_scale": kl_bnn_scale,
        "perceptual_loss": perceptual_loss,
        "perceptual_scale": perceptual_scale,
        "total_loss": total_loss,
    }

    write_header = not os.path.exists(output_file)

    with open(output_file, 'a') as f:
        if write_header:
            f.write("\t".join(stats.keys()) + "\n")
        f.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in stats.values()) + "\n")
