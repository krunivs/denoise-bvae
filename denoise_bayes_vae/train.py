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
    metrics_curve_path = os.path.join(model_dir, 'metric_curve.png')
    train_report_file = ''.join([report_path_prefix, 'train.tsv'])
    validation_report_file = ''.join([report_path_prefix, 'validation.tsv'])

    if not os.path.isdir(os.path.dirname(report_path_prefix)):
        raise NotADirectoryError(f'Not found {os.path.dirname(report_path_prefix)} directory to save report.')

    ''' statistics '''
    kl_z_scales, stft_scales, kl_bnn_scales, perceptual_scales = [], [], [], []
    train_loss_avgs, validation_loss_avgs = [], []
    train_mse_loss_avgs, validation_mse_loss_avgs = [], []
    train_kl_z_loss_avgs, validation_kl_z_loss_avgs = [], []
    train_stft_loss_avgs, validation_stft_loss_avgs = [], []
    train_perceptual_loss_avgs, validation_perceptual_loss_avgs = [], []
    train_kl_bnn_loss_avgs, validation_kl_bnn_loss_avgs = [], []
    train_stoi_avgs, validation_stoi_avgs = [], []
    train_snr_avgs, validation_snr_avgs = [], []
    train_pesq_avgs, validation_pesq_avgs = [], []

    ''' Early stopping and loss tracking '''
    best_val_loss = float('inf')
    patience_counter = 0

    ''' Initial weight scalars '''
    # Scaling factor for KL divergence on latent z
    kl_z_scale = 1e-3
    # Scaling factor for STFT loss
    stft_scale = 0.1
    # Scaling factor for kl_bnn_loss
    kl_bnn_scale = 1e-3
    # Scaling factor for perceptual_scale
    perceptual_scale = 0.3

    logger.info(f"Training started: "
                f"kl_z_scale={kl_z_scale}, "
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
        total_mse_loss = 0
        total_stft_loss = 0
        total_kl_z_loss = 0
        total_kl_bnn_loss = 0
        total_perceptual_loss = 0
        stoi_scores = []
        snr_scores = []
        pesq_scores = []

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

            with torch.no_grad():
                ''' Calculate STOI, PESQ, SNR '''
                stoi, pesq, snr = compute_metrics(clean, recon)

                ''' Calculate Reconstructed output cosine similarity mean '''
                batch_size = recon.size(0)
                recon_flat = recon.view(batch_size, -1)
                cosine_sim = F.cosine_similarity(recon_flat.unsqueeze(1), recon_flat.unsqueeze(0), dim=-1)
                upper_tri = cosine_sim[torch.triu(torch.ones_like(cosine_sim), diagonal=1) > 0]
                recon_cos_sim_mean = upper_tri.mean().item()

                z_std_value = z.std().item()
                mu_std_value = mu.std().item()
                logvar_std_value = logvar.std().item()
                recon_std_value = recon.std().item()
                alpha_value = model.decoder.alpha.item()
                mse_loss_value = mse_loss.item()
                stft_loss_value = stft_loss.item()
                kl_z_loss_value = kl_z_loss.item()
                kl_bnn_loss_value = kl_bnn_loss.item()
                perceptual_loss_value = perceptual_loss.item()
                total_loss_value = total_elbo_loss.item()

                total_mse_loss += mse_loss_value
                total_stft_loss += stft_loss_value
                total_perceptual_loss += perceptual_loss_value
                total_kl_z_loss += kl_z_loss_value
                total_kl_bnn_loss += kl_bnn_loss_value
                total_train_loss += total_loss_value

                if stoi is not None:
                    stoi_scores.append(stoi)
                if pesq is not None:
                    pesq_scores.append(pesq)
                if snr is not None:
                    snr_scores.append(snr)
                '''
                # Collapse warnings
                  Warning Conditions:
                    - z_std < 0.01: posterior collapse
                    - mu_std < 0.01: posterior collapse
                    - logvar_std < 0.01: prior collapse
                    - recon_std < 1e-3: decoder collapsed to near-constant output
                    - recon_cos_sim > 0.99: decoder output repetition (mode collapse)            
                '''
                if z_std_value < 0.01:
                    logger.warning(f"[Training][CollapseWarning] z_std too small: {z_std_value:.4f}")
                if mu_std_value < 0.01:
                    logger.warning(f"[Training][CollapseWarning] mu_std too small: {mu_std_value:.4f}")
                if logvar_std_value < 0.01:
                    logger.warning(f"[Training][CollapseWarning] logvar_std too small: {logvar_std_value:.4f}")
                if recon_std_value < 1e-3:
                    logger.warning(f"[Training][CollapseWarning] recon_x collapsed: std={recon_std_value:.5f}")
                if recon_cos_sim_mean > 0.99:
                    logger.warning(f"[Training][CollapseWarning] recon_x repeated pattern: cosine_sim={recon_cos_sim_mean:.4f}")

                # Generate train report
                if is_report_provided:
                    generate_train_report(
                        mu=mu,
                        logvar=logvar,
                        z=z,
                        recon=recon,
                        recon_cos_sim_mean=recon_cos_sim_mean,
                        alpha=alpha_value,
                        stoi=stoi,
                        pesq=pesq,
                        snr=snr,
                        epoch=epoch + 1,
                        kl_z_scale=kl_z_scale,
                        stft_scale=stft_scale,
                        kl_bnn_scale=kl_bnn_scale,
                        perceptual_scale=perceptual_scale,
                        mse_loss=mse_loss_value,
                        stft_loss=stft_loss_value,
                        kl_z_loss=kl_z_loss_value,
                        kl_bnn_loss=kl_bnn_loss_value,
                        perceptual_loss=perceptual_loss_value,
                        total_loss=total_loss_value,
                        output_file=train_report_file)

        # plot data for entire epochs
        n_iterations = len(train_loader)
        train_loss_avgs.append(total_train_loss / n_iterations)
        train_mse_loss_avgs.append(total_mse_loss / n_iterations)
        train_stft_loss_avgs.append(total_stft_loss / n_iterations)
        train_perceptual_loss_avgs.append(total_perceptual_loss / n_iterations)
        train_kl_z_loss_avgs.append(total_kl_z_loss / n_iterations)
        train_kl_bnn_loss_avgs.append(total_kl_bnn_loss / n_iterations)
        avg_stoi, avg_pesq, avg_snr = None, None, None
        if len(stoi_scores) > 0:
            avg_stoi = sum(stoi_scores) / len(stoi_scores)
        if len(snr_scores) > 0:
            avg_snr = sum(snr_scores) / len(snr_scores)
        if len(pesq_scores) > 0:
            avg_pesq = sum(pesq_scores) / len(pesq_scores)
        train_stoi_avgs.append(avg_stoi)
        train_snr_avgs.append(avg_snr)
        train_pesq_avgs.append(avg_pesq)
        kl_z_scales.append(kl_z_scale)
        stft_scales.append(stft_scale)
        kl_bnn_scales.append(kl_bnn_scale)
        perceptual_scales.append(perceptual_scale)

        # Validation loop
        model.eval()
        model.decoder.use_skip = False
        total_val_loss = 0
        total_mse_loss = 0
        total_stft_loss = 0
        total_kl_z_loss = 0
        total_kl_bnn_loss = 0
        total_perceptual_loss = 0
        snr_scores = []
        stoi_scores = []
        pesq_scores = []

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
                    perceptual_scale=perceptual_scale,
                    epoch=epoch + 1,
                    auto_adjust_kl=False,
                    auto_adjust_stft=False,
                    auto_adjust_kl_bnn=False)

                total_elbo_loss = loss_outputs[0]
                mse_loss = loss_outputs[1]
                stft_loss = loss_outputs[2]
                kl_z_loss = loss_outputs[3]
                perceptual_loss = loss_outputs[4]

                ''' Calculate STOI, PESQ, SNR '''
                stoi, pesq, snr = compute_metrics(clean, recon)

                ''' Calculate Reconstructed output cosine similarity mean '''
                batch_size = recon.size(0)
                recon_flat = recon.view(batch_size, -1)
                cosine_sim = F.cosine_similarity(recon_flat.unsqueeze(1), recon_flat.unsqueeze(0), dim=-1)
                upper_tri = cosine_sim[torch.triu(torch.ones_like(cosine_sim), diagonal=1) > 0]
                recon_cos_sim_mean = upper_tri.mean().item()
                z_std_value = z.std().item()
                mu_std_value = mu.std().item()
                logvar_std_value = logvar.std().item()
                recon_std_value = recon.std().item()
                alpha_value = model.decoder.alpha.item()

                total_loss_value = total_elbo_loss.item()
                mse_loss_value = mse_loss.item()
                stft_loss_value = stft_loss.item()
                perceptual_loss_value = perceptual_loss.item()
                kl_z_loss_value = kl_z_loss.item()
                kl_bnn_loss_value = kl_bnn_loss.item()

                if stoi is not None:
                    stoi_scores.append(stoi)
                if pesq is not None:
                    pesq_scores.append(pesq)
                if snr is not None:
                    snr_scores.append(snr)

                total_val_loss += total_loss_value
                total_mse_loss += mse_loss_value
                total_stft_loss += stft_loss_value
                total_perceptual_loss += perceptual_loss_value
                total_kl_z_loss += kl_z_loss_value
                total_kl_bnn_loss += kl_bnn_loss_value

                if z_std_value < 0.01:
                    logger.warning(f"[Validation][CollapseWarning] z_std too small: {z_std_value:.4f}")
                if mu_std_value < 0.01:
                    logger.warning(f"[Validation][CollapseWarning] mu_std too small: {mu_std_value:.4f}")
                if logvar_std_value < 0.01:
                    logger.warning(f"[Validation][CollapseWarning] logvar_std too small: {logvar_std_value:.4f}")
                if recon_std_value < 1e-3:
                    logger.warning(f"[Validation][CollapseWarning] recon_x collapsed: std={recon_std_value:.5f}")
                if recon_cos_sim_mean > 0.99:
                    logger.warning(f"[Validation][CollapseWarning] recon_x repeated pattern: cosine_sim={recon_cos_sim_mean:.4f}")

                # Generate validation report
                if is_report_provided:
                    generate_validation_report(
                        epoch=epoch + 1,
                        mu=mu,
                        logvar=logvar,
                        z=z,
                        recon=recon,
                        recon_cos_sim_mean=recon_cos_sim_mean,
                        alpha=alpha_value,
                        stoi=stoi,
                        pesq=pesq,
                        snr=snr,
                        mse_loss=mse_loss_value,
                        stft_loss=stft_loss_value,
                        stft_scale=stft_scale,
                        kl_z_loss=kl_z_loss_value,
                        kl_z_scale=kl_z_scale,
                        kl_bnn_loss=kl_bnn_loss_value,
                        kl_bnn_scale=kl_bnn_scale,
                        perceptual_loss=perceptual_loss_value,
                        perceptual_scale=perceptual_scale,
                        total_loss=total_loss_value,
                        output_file=validation_report_file)

        iterations = len(val_loader)
        validation_loss_avgs.append(total_val_loss / iterations)

        avg_stoi, avg_pesq, avg_snr = None, None, None
        if len(stoi_scores) > 0:
            avg_stoi = sum(stoi_scores) / len(stoi_scores)
        if len(pesq_scores) > 0:
            avg_pesq = sum(pesq_scores) / len(pesq_scores)
        if len(snr_scores) > 0:
            avg_snr = sum(snr_scores) / len(snr_scores)
        avg_mse_loss = total_mse_loss / iterations
        avg_stft_loss = total_stft_loss / iterations
        avg_perceptual_loss = total_perceptual_loss / iterations
        avg_kl_z_loss = total_kl_z_loss / iterations
        avg_kl_bnn_loss = total_kl_bnn_loss / iterations

        # plot data for entire epochs
        validation_mse_loss_avgs.append(avg_mse_loss)
        validation_stft_loss_avgs.append(avg_stft_loss)
        validation_perceptual_loss_avgs.append(avg_perceptual_loss)
        validation_kl_z_loss_avgs.append(avg_kl_z_loss)
        validation_kl_bnn_loss_avgs.append(avg_kl_bnn_loss)
        validation_stoi_avgs.append(avg_stoi)
        validation_snr_avgs.append(avg_snr)
        validation_pesq_avgs.append(avg_pesq)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} [{epoch_time:.2f}s] "
                    f"Train Loss={train_loss_avgs[-1]:.4f} "
                    f"Val Loss={validation_loss_avgs[-1]:.4f}")

        # Early stopping and model saving
        if validation_loss_avgs[-1] < best_val_loss:
            best_val_loss = validation_loss_avgs[-1]
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
    plt.plot(range(1, len(train_loss_avgs) + 1), train_loss_avgs, label='Train Loss')
    plt.plot(range(1, len(validation_loss_avgs) + 1), validation_loss_avgs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_curve_path)
    logger.info(f"Loss curve saved at {loss_curve_path}")

    # Save KL annealing scale curves
    plt.figure(figsize=(12, 24))  # 높이 늘림 (5개 subplot)

    # 1. MSE Loss (Train & Valid)
    plt.subplot(5, 1, 1)
    plt.plot(train_mse_loss_avgs, label='Train MSE Loss', marker='o')
    plt.plot(validation_mse_loss_avgs, label='Val MSE Loss', marker='o')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # 2. KL Z Loss (Train & Valid)
    plt.subplot(5, 1, 2)
    plt.plot(train_kl_z_loss_avgs, label='Train KL Z Loss', marker='o')
    plt.plot(validation_kl_z_loss_avgs, label='Val KL Z Loss', marker='o')
    plt.plot(kl_z_scales, label='KL Z Scale', marker='x')
    plt.title('KL Z Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # 3. STFT Loss (Train & Valid)
    plt.subplot(5, 1, 3)
    plt.plot(train_stft_loss_avgs, label='Train STFT Loss', marker='o')
    plt.plot(validation_stft_loss_avgs, label='Val STFT Loss', marker='o')
    plt.plot(stft_scales, label='STFT Scale', marker='x')
    plt.title('STFT Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # 4. Perceptual Loss (Train & Valid)
    plt.subplot(5, 1, 4)
    plt.plot(train_perceptual_loss_avgs, label='Train Perceptual Loss', marker='o')
    plt.plot(validation_perceptual_loss_avgs, label='Val Perceptual Loss', marker='o')
    plt.plot(perceptual_scales, label='Perceptual Scale', marker='x')
    plt.title('Perceptual Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # 5. KL BNN Loss (Train & Valid)
    plt.subplot(5, 1, 5)
    plt.plot(train_kl_bnn_loss_avgs, label='Train KL BNN Loss', marker='o')
    plt.plot(validation_kl_bnn_loss_avgs, label='Val KL BNN Loss', marker='o')
    plt.plot(kl_bnn_scales, label='KL BNN Scale', marker='x')
    plt.title('KL BNN Loss vs. Scale')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # Save
    plt.tight_layout()
    plt.savefig(scale_curve_path)
    logger.info(f"Scale curve saved at {scale_curve_path}")

    # Save STOI, SNR, PESQ curve
    plt.figure(figsize=(12, 14))  # 3개 subplot

    # 1. STOI
    plt.subplot(3, 1, 1)
    plt.plot(train_stoi_avgs, label='Train STOI', marker='o')
    plt.plot(validation_stoi_avgs, label='Validation STOI', marker='x')
    plt.title('STOI (Train & Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('STOI')
    plt.grid(True)
    plt.legend()

    # 2. SNR
    plt.subplot(3, 1, 2)
    plt.plot(train_snr_avgs, label='Train SNR', marker='o')
    plt.plot(validation_snr_avgs, label='Validation SNR', marker='x')
    plt.title('SNR (Train & Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    plt.legend()

    # 3. PESQ
    plt.subplot(3, 1, 3)
    plt.plot(train_pesq_avgs, label='Train PESQ', marker='o')
    plt.plot(validation_pesq_avgs, label='Validation PESQ', marker='x')
    plt.title('PESQ (Train & Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('PESQ')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(metrics_curve_path)
    logger.info(f"Metrics curve saved at {metrics_curve_path}")
