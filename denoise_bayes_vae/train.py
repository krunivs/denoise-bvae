# denoise_bayes_vae/train.py
# -*- encoding: utf-8 -*-

import os
import time
import torch
import matplotlib
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.audio_qc import compute_metrics
from utils.logger import Logger
from denoise_bayes_vae.loss import elbo_loss
from utils.report import generate_train_report, generate_validation_report

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = Logger().get_logger()

def avg_pair(stats):
    m_values, s_values = zip(*stats)
    return sum(m_values) / len(m_values), sum(s_values) / len(s_values)

def train(model: nn.Module,
          optimizer: optim.Optimizer,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: torch.device,
          model_save_path: str,
          autodiff: bool = False,
          epochs: int = 100,
          patience: int = 5,
          report_path_prefix: str = None,
          is_report_provided: bool = False) -> None:
    """
    Train a Bayesian VAE model using ELBO loss with dynamic scaling for KL, STFT, and BNN loss terms.

    Generates both training and validation reports and saves learning curves.

    Args:
        model (nn.Module): The Bayesian VAE model to train.
        optimizer (optim.Optimizer): Optimizer instance (e.g., Adam).
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        device (torch.device): Device to run training on (e.g., cuda or cpu).
        model_save_path (str): Path to save the best model.
        autodiff (bool): If True, use manual autograd.grad; otherwise, use backward().
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        report_path_prefix (str): Prefix path to save train/validation reports.
        is_report_provided (bool): True - provide report
    """
    # Filepath config
    model_dir = os.path.dirname(model_save_path)
    loss_curve_path = os.path.join(model_dir, 'loss_curve.png')
    scale_curve_path = os.path.join(model_dir, 'kl_anneal_scale_curve.png')
    train_report_file = ''.join([report_path_prefix, 'train.tsv'])
    validation_report_file = ''.join([report_path_prefix, 'validation.tsv'])

    if not os.path.isdir(os.path.dirname(report_path_prefix)):
        raise NotADirectoryError(f'Not found {os.path.dirname(report_path_prefix)} directory to save report.')

    # Early stopping and loss tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, validation_losses = [], []
    kl_z_scales, stft_scales, kl_bnn_scales = [], [], []
    kl_z_losses, stft_losses, kl_bnn_losses = [], [], []

    # Initial weight scalars
    kl_z_scale = 0.1        # Scaling factor for KL divergence on latent z
    stft_scale = 1.0        # Scaling factor for STFT loss
    kl_bnn_scale = 0.001    #  Scaling factor for kl_bnn_loss
    perceptual_scale = 0.05 #  Scaling factor for perceptual_scale

    logger.info(f"Training started: kl_z_scale={kl_z_scale}, "
                f"stft_scale={stft_scale}, "
                f"kl_bnn_scale={kl_bnn_scale}, "
                f"perceptual_scale={perceptual_scale}")

    start_time = time.time()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        # Training loop
        epoch_start_time = time.time()
        model.train()
        model.decoder.use_skip = True
        total_train_loss = 0

        # batch
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            recon, mu, logvar, z = model(noisy)
            kl_bnn_loss = model.kl_loss() * kl_bnn_scale

            loss_outputs = elbo_loss(
                recon, clean, mu, logvar,
                kl_z_scale=kl_z_scale,
                kl_bnn_loss=kl_bnn_loss,
                kl_bnn_scale=kl_bnn_scale,
                perceptual_scale=perceptual_scale,
                stft_scale=stft_scale,
                epoch=epoch + 1)

            total_elbo_loss = loss_outputs[0]
            mse_loss = loss_outputs[1]
            stft_loss = loss_outputs[2]
            kl_z_loss = loss_outputs[3]
            perceptual_loss = loss_outputs[4]
            kl_z_scale = loss_outputs[5]
            stft_scale = loss_outputs[6]
            kl_bnn_scale = loss_outputs[7]
            perceptual_scale = loss_outputs[8]

            if autodiff:
                grads = torch.autograd.grad(total_elbo_loss, model.parameters())
                for p, g in zip(model.parameters(), grads):
                    if p.requires_grad and g is not None:
                        p.grad = g
            else:
                total_elbo_loss.backward()

            optimizer.step()
            total_train_loss += total_elbo_loss.item()
            batch_size = recon.size(0)
            recon_flat = recon.view(batch_size, -1)

            with torch.no_grad():
                cosine_sim = F.cosine_similarity(recon_flat.unsqueeze(1), recon_flat.unsqueeze(0), dim=-1)
                upper_tri = cosine_sim[torch.triu(torch.ones_like(cosine_sim), diagonal=1) > 0]
                recon_cos_sim_mean = upper_tri.mean().item()

            '''
            # Collapse warnings
              Warning Conditions:
                - z_std < 0.01: posterior collapse
                - mu_std < 0.01: posterior collapse
                - logvar_std < 0.01: prior collapse
                - recon_std < 1e-3: decoder collapsed to near-constant output
                - recon_cos_sim > 0.99: decoder output repetition (mode collapse)            
            '''
            if z.std().item() < 0.01:
                logger.warning(f"[CollapseWarning] z_std too small: {z.std().item():.4f}")
            if mu.std().item() < 0.01:
                logger.warning(f"[CollapseWarning] mu_std too small: {mu.std().item():.4f}")
            if logvar.std().item() < 0.01:
                logger.warning(f"[CollapseWarning] logvar_std too small: {logvar.std().item():.4f}")
            if recon.std().item() < 1e-3:
                logger.warning(f"[CollapseWarning] recon_x collapsed: std={recon.std().item():.5f}")
            if recon_cos_sim_mean > 0.99:
                logger.warning(f"[CollapseWarning] recon_x repeated pattern: cosine_sim={recon_cos_sim_mean:.4f}")

            # Generate train report
            if is_report_provided:
                generate_train_report(
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    recon=recon,
                    recon_cos_sim_mean=recon_cos_sim_mean,
                    alpha=model.decoder.alpha.item(),
                    skip_z=model.decoder.skip_z(z).norm().item(),
                    epoch=epoch + 1,
                    kl_z_scale=kl_z_scale,
                    stft_scale=stft_scale,
                    kl_bnn_scale=kl_bnn_scale,
                    perceptual_scale=perceptual_scale,
                    mse_loss=mse_loss.item(),
                    stft_loss=stft_loss.item(),
                    kl_z_loss=kl_z_loss.item(),
                    kl_bnn_loss=kl_bnn_loss.item(),
                    perceptual_loss=perceptual_loss.item(),
                    total_loss=total_elbo_loss.item(),
                    output_file=train_report_file)

        # statistic data
        train_losses.append(total_train_loss / len(train_loader))
        kl_z_scales.append(kl_z_scale)
        stft_scales.append(stft_scale)
        kl_bnn_scales.append(kl_bnn_scale)

        # Validation loop
        model.eval()
        model.decoder.use_skip = False
        total_val_loss = 0
        snr_scores, stoi_scores, pesq_scores = [], [], []
        mu_stats, logvar_stats, z_stats, recon_stats = [], [], [], []
        mse_losses, stft_losses, kl_z_losses, kl_bnn_losses, perceptual_losses, total_losses = [], [], [], [], [], []

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                recon, mu, logvar, z = model(noisy)
                kl_bnn_loss = model.kl_loss() * kl_bnn_scale

                loss_outputs = elbo_loss(
                    recon, clean, mu, logvar,
                    kl_z_scale=kl_z_scale,
                    kl_bnn_loss=kl_bnn_loss,
                    kl_bnn_scale=kl_bnn_scale,
                    stft_scale=stft_scale,
                    epoch=epoch + 1,
                    auto_adjust_kl=False,
                    auto_adjust_stft=False,
                    auto_adjust_kl_bnn=False)

                total_elbo_loss = loss_outputs[0]
                total_val_loss += total_elbo_loss.item()

                if is_report_provided:
                    mse_loss = loss_outputs[1]
                    stft_loss = loss_outputs[2]
                    kl_z_loss = loss_outputs[3]
                    perceptual_loss = loss_outputs[4]
                    mu_stats.append((mu.mean().item(), mu.std().item()))
                    logvar_stats.append((logvar.mean().item(), logvar.std().item()))
                    z_stats.append((z.mean().item(), z.std().item()))
                    recon_stats.append((recon.mean().item(), recon.std().item()))
                    mse_losses.append(mse_loss.item())
                    stft_losses.append(stft_loss.item())
                    kl_z_losses.append(kl_z_loss.item())
                    kl_bnn_losses.append(kl_bnn_loss.item())
                    perceptual_losses.append(perceptual_loss.item())
                    total_losses.append(total_elbo_loss.item())

                for i in range(noisy.size(0)):
                    recon_i, clean_i = recon[i], clean[i]
                    stoi, pesq, snr = compute_metrics(clean_i, recon_i)
                    snr_scores.append(snr)
                    stoi_scores.append(stoi)
                    pesq_scores.append(pesq)

        validation_losses.append(total_val_loss / len(val_loader))
        kl_z_losses.append(sum(kl_z_losses) / len(kl_z_losses))
        stft_losses.append(sum(stft_losses) / len(stft_losses))
        kl_bnn_losses.append(sum(kl_bnn_losses) / len(kl_bnn_losses))

        # Generate validation report
        if is_report_provided:
            generate_validation_report(
                epoch=epoch + 1,
                mu_mean=avg_pair(mu_stats)[0],
                mu_std=avg_pair(mu_stats)[1],
                logvar_mean=avg_pair(logvar_stats)[0],
                logvar_std=avg_pair(logvar_stats)[1],
                z_mean=avg_pair(z_stats)[0],
                z_std=avg_pair(z_stats)[1],
                recon_mean=avg_pair(recon_stats)[0],
                recon_std=avg_pair(recon_stats)[1],
                avg_stoi=sum(stoi_scores) / len(stoi_scores),
                avg_pesq=sum(pesq_scores) / len(pesq_scores),
                avg_snr=sum(snr_scores) / len(snr_scores),
                mse_loss=sum(mse_losses) / len(mse_losses),
                stft_loss=sum(stft_losses) / len(stft_losses),
                kl_z_loss=sum(kl_z_losses) / len(kl_z_losses),
                kl_bnn_loss=sum(kl_bnn_losses) / len(kl_bnn_losses),
                perceptual_loss=sum(perceptual_losses) / len(perceptual_losses),
                total_loss=sum(total_losses) / len(total_losses),
                output_file=validation_report_file)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} [{epoch_time:.2f}s] "
                    f"Train Loss={train_losses[-1]:.4f} "
                    f"Val Loss={validation_losses[-1]:.4f}")

        # Early stopping and model saving
        if validation_losses[-1] < best_val_loss:
            best_val_loss = validation_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Best model saved at {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")

    # Save loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_curve_path)
    logger.info(f"Loss curve saved at {loss_curve_path}")

    # Save KL annealing scale curves
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plt.plot(kl_z_losses, label='KL Z Loss', marker='o')
    plt.plot(kl_z_scales, label='KL Z Scale', marker='x')
    plt.title('KL Z Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(stft_losses, label='STFT Loss', marker='o')
    plt.plot(stft_scales, label='STFT Scale', marker='x')
    plt.title('STFT Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(kl_bnn_losses, label='KL BNN Loss', marker='o')
    plt.plot(kl_bnn_scales, label='KL BNN Scale', marker='x')
    plt.title('KL BNN Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(scale_curve_path)
    logger.info(f"Scale curve saved at {scale_curve_path}")

