# utils/report.py
# -*- encoding: utf-8 -*-

import os

def generate_train_report(
        mu,
        logvar,
        z,
        recon,
        recon_cos_sim_mean,
        alpha,
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
        "mu_mean": mu.mean().item(),
        "mu_std": mu.std().item(),
        "logvar_mean": logvar.mean().item(),
        "logvar_std": logvar.std().item(),
        "z_mean": z.mean().item(),
        "z_std": z.std().item(),
        "recon_mean": recon.mean().item(),
        "recon_std": recon.std().item(),
        "recon_cos_sim": recon_cos_sim_mean,
        "alpha": alpha,
        "kl_z_scale": kl_z_scale,
        "stft_scale": stft_scale,
        "kl_bnn_scale": kl_bnn_scale,
        "perceptual_scale": perceptual_scale,
        "mse_loss": float(mse_loss) if mse_loss is not None else -1,
        "stft_loss": float(stft_loss) if stft_loss is not None else -1,
        "kl_z_loss": float(kl_z_loss) if kl_z_loss is not None else -1,
        "kl_bnn_loss": float(kl_bnn_loss) if kl_bnn_loss is not None else -1,
        "perceptual_loss": perceptual_loss if perceptual_loss is not None else -1,
        "total_loss": float(total_loss) if total_loss is not None else -1
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
        mu_mean: float,
        mu_std: float,
        logvar_mean: float,
        logvar_std: float,
        z_mean: float,
        z_std: float,
        recon_mean: float,
        recon_std: float,
        avg_stoi: float,
        avg_pesq: float,
        avg_snr: float,
        mse_loss: float = -1.0,
        stft_loss: float = -1.0,
        kl_z_loss: float = -1.0,
        kl_bnn_loss: float = -1.0,
        perceptual_loss: float = -1.0,
        total_loss: float = -1.0,
        output_file: str = None):
    """
    Generate and log validation statistics to monitor model health and performance.

    This function records numerical indicators of latent variable behavior and reconstruction quality
    during validation. It is designed to identify signs of posterior/prior collapse and track
    loss dynamics across epochs. The format is aligned with `generate_train_report()` for consistency.

    Args:
        epoch (int): Current epoch number.
        mu_mean (float): Mean of the latent variable means.
        mu_std (float): Std of the latent variable means.
        logvar_mean (float): Mean of log-variance from encoder.
        logvar_std (float): Std of log-variance from encoder.
        z_mean (float): Mean of sampled latent variable z.
        z_std (float): Std of sampled latent variable z.
        recon_mean (float): Mean of reconstructed output.
        recon_std (float): Std of reconstructed output.
        avg_stoi (float): Average Short-Time Objective Intelligibility (STOI) score.
        avg_pesq (float): Average PESQ score.
        avg_snr (float): Average Signal-to-Noise Ratio.
        mse_loss (float, optional): MSE loss value (default: -1.0).
        stft_loss (float, optional): STFT loss value (default: -1.0).
        kl_z_loss (float, optional): Latent KL divergence loss (default: -1.0).
        kl_bnn_loss (float, optional): KL loss from Bayesian layers (default: -1.0).
        perceptual_loss (float, optional): MFCC-based Perceptual loss value (default: -1.0).
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
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "logvar_mean": logvar_mean,
        "logvar_std": logvar_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "recon_mean": recon_mean,
        "recon_std": recon_std,
        "stoi": avg_stoi,
        "pesq": avg_pesq,
        "snr": avg_snr,
        "mse_loss": mse_loss,
        "stft_loss": stft_loss,
        "kl_z_loss": kl_z_loss,
        "kl_bnn_loss": kl_bnn_loss,
        "perceptual_loss": perceptual_loss,
        "total_loss": total_loss,
    }

    write_header = not os.path.exists(output_file)

    with open(output_file, 'a') as f:
        if write_header:
            f.write("\t".join(stats.keys()) + "\n")
        f.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in stats.values()) + "\n")
