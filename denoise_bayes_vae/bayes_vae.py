# denoise_bayes_vae/bayes_vae.py
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from denoise_bayes_vae.sample import sample_latent
from utils.logger import Logger

logger = Logger().get_logger()


class BayesianLinear(nn.Module):
    """
    Bayesian fully connected layer with Gaussian or Student-t posterior sampling.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        dist_type (str): Distribution type ('gaussian' or 'student-t').
        df (float): Degrees of freedom for student-t (if used).
    """
    def __init__(self, in_features, out_features, dist_type='gaussian', df=3.0):
        super().__init__()
        self.dist_type = dist_type
        self.df = df
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.01))
        self.logvar_w = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5.0, 0.1))
        self.mu_b = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.logvar_b = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

    def forward(self, x):
        w = sample_latent(self.mu_w, self.logvar_w, self.dist_type, self.df)
        b = sample_latent(self.mu_b, self.logvar_b, self.dist_type, self.df)
        return F.linear(x, w, b)

    def kl_loss(self):
        kl_w = -0.5 * torch.sum(1 + self.logvar_w - self.mu_w.pow(2) - self.logvar_w.exp())
        kl_b = -0.5 * torch.sum(1 + self.logvar_b - self.mu_b.pow(2) - self.logvar_b.exp())
        num_params = self.mu_w.numel() + self.mu_b.numel()
        return (kl_w + kl_b) / num_params


class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers and ReLU activation.

    Args:
        dim (int): Dimension of the input and output.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)


class Postnet(nn.Module):
    """
    Post-processing network used in the decoder of a Bayesian VAE.

    This module refines the initial output of the decoder by capturing residual
    artifacts or patterns that were not adequately reconstructed in the main decoding path.
    It consists of multiple fully connected layers with non-linear activations and dropout
    for regularization.

    Args:
        dim (int): The dimensionality of both the input and output. This should match the
                   output dimension of the decoder and is typically equal to the number
                   of audio samples per chunk (e.g., 16000 for 1 second at 16kHz).

    Architecture:
        Linear(dim → dim) → ReLU → Dropout(0.2) →
        Linear(dim → dim) → ReLU → Dropout(0.2) →
        Linear(dim → dim)

    Returns:
        Tensor: Refined waveform of shape (B, dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class ConvPostnet(nn.Module):
    def __init__(self, input_dim: int, channels: int = 64, kernel_size: int = 5, num_layers: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, channels)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))
        self.conv_net = nn.Sequential(*layers)
        self.output_proj = nn.Linear(channels, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        x = self.input_proj(x).unsqueeze(1)   # → [B, 1, C]
        x = x.transpose(1, 2)                 # → [B, C, 1]
        x = self.conv_net(x)                 # → [B, C, 1]
        x = x.transpose(1, 2).squeeze(1)      # → [B, C]
        x = self.output_proj(x)              # → [B, T]

        return x

class Encoder(nn.Module):
    """
    Encoder for Bayesian VAE(Variational AutoEncoder)
    """
    def __init__(self, input_dim, latent_dim, dist_type='gaussian', df=3.0):
        super().__init__()
        self.dist_type = dist_type
        self.df = df

        self.fc1 = BayesianLinear(input_dim, 512, dist_type, df)
        self.fc2 = BayesianLinear(512, 512, dist_type, df)
        self.fc3 = BayesianLinear(512, 256, dist_type, df)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # initialize weight for stable learning
        nn.init.kaiming_normal_(self.fc_mu.weight, nonlinearity='linear')  # more variance
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.normal_(self.fc_logvar.weight, std=0.1)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))     # input_dim → 512
        x = F.relu(self.fc2(x))     # 512 → 512
        x = F.relu(self.fc3(x))     # 512 → 256
        mu = self.fc_mu(x)          # 256 → latent_dim
        logvar = self.fc_logvar(x)  # 256 → latent_dim

        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            logger.warning("NaN in mu/logvar at encoder output")

        return mu, logvar

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, dist_type='gaussian', df=3.0, use_skip=True):
        super().__init__()
        self.dist_type = dist_type
        self.df = df
        self.use_skip = use_skip
        self.output_dim = output_dim

        self.fc1 = BayesianLinear(latent_dim, 256, dist_type, df)
        self.fc2 = BayesianLinear(256 + 128, 512, dist_type, df)

        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU()
        )
        self.z_deep = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.skip_z = nn.Linear(latent_dim, output_dim)
        self.resblock = ResidualBlock(768)

        self.output_layer = nn.Sequential(
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
        # self.postnet = ConvPostnet(output_dim)
        self.postnet = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=9, padding=4)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gate

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        z_proj = self.z_proj(z)
        z_deep = self.z_deep(z)

        z1 = F.dropout(F.relu(self.fc1(z)), p=0.15, training=self.training)
        z1_concat = torch.cat([z1, z_proj], dim=-1)
        z2 = F.relu(self.fc2(z1_concat))

        fused = torch.cat([z2, z_proj, z_deep], dim=-1)
        res = self.resblock(fused)
        res = F.dropout(res, p=0.25, training=self.training)

        out = self.output_layer(res)        # [B, output_dim]
        # refined = self.postnet(out)         # Postnet now operates directly on output
        refined = self.postnet(out.unsqueeze(1)).squeeze(1)  # Conv1D postnet 적용

        alpha = torch.sigmoid(self.alpha)
        if self.training and (alpha.item() < 0.1 or alpha.item() > 0.9):
            logger.warning(f"[Gating] alpha extreme: alpha={alpha.item():.3f}")

        # combined = out + alpha * (refined - out)  # Residual로 처리하여 collapse 완화
        combined = (1 - alpha) * out + alpha * refined  # gate residual

        if self.use_skip:
            # combined += 0.5 * self.skip_z(z)
            combined += 0.7 * self.skip_z(z)  # skip 비중 상향

        if torch.isnan(combined).any():
            logger.error("NaN detected in final decoder output!")

        return combined

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()


class BayesianVAE(nn.Module):
    """
    Bayesian VAE(Variational AutoEncoder) Model
    """
    def __init__(self, input_dim, latent_dim, dist_type='gaussian', df=3.0):
        """
        :param input_dim: (int) input dimension
        :param latent_dim: (int) latent dimension
        :param dist_type: (str) probability distribution type (either 'gaussian' or 'student-t')
        :param df: (float) degrees of freedom
        """
        super().__init__()
        self.dist_type = dist_type
        self.df = df
        self.encoder = Encoder(input_dim, latent_dim, dist_type, df)
        self.decoder = Decoder(latent_dim, input_dim, dist_type, df)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = sample_latent(mu, logvar, self.dist_type, self.df)
        z = F.dropout(z, p=0.1, training=self.training)

        if torch.isnan(z).any() or z.std() > 10.0:
            logger.warning(f"[z warning] z has extreme std: {z.std().item():.4f}")

        if mu.std() < 0.01 or logvar.std() < 0.01:
            logger.warning(f"[Collapse] mu.std={mu.std():.4f}, logvar.std={logvar.std():.4f}")

        recon_x = self.decoder(z)

        if torch.isnan(z).any():
            logger.error("NaN detected in latent z!")
        if torch.isnan(mu).any():
            logger.error("NaN detected in mu!")
        if torch.isnan(logvar).any():
            logger.error("NaN detected in logvar!")

        return recon_x, mu, logvar, z

    def kl_loss(self):
        return self.encoder.kl_loss() + self.decoder.kl_loss()