# denoise_bayes_vae/bayes_vae.py
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from denoise_bayes_vae.sample import sample_latent
from utils.logger import Logger
import torch.utils.checkpoint as cp

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
    def __init__(self, latent_dim, dist_type='gaussian', df=3.0):
        super().__init__()
        self.dist_type = dist_type
        self.df = df

        # Conv1D 기반 feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),  # [B, 1, T] → [B, 64, T]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),  # [B, 64, T] → [B, 128, T/2]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),  # [B, 128, T/2] → [B, 256, T/4]
            nn.ReLU()
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # → [B, 256, 1]
        self.linear_in = nn.Linear(256, 256)

        # Fully connected Bayesian latent mapping
        self.fc1 = BayesianLinear(256, 256, dist_type, df)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # initialize weight for stable learning
        nn.init.kaiming_normal_(self.fc_mu.weight, nonlinearity='linear')  # more variance
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.normal_(self.fc_logvar.weight, std=0.1)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, T] → [B, 1, T]
        x = self.conv(x)    # [B, 256, T/4]
        x = self.temporal_pool(x).squeeze(-1)  # [B, 256]
        x = F.relu(self.linear_in(x))
        x = F.relu(self.fc1(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        mu = F.layer_norm(mu, mu.shape[1:])  # normalization 추가(mu에 대한 정규화는 posterior collapse 방지에 효과적)
        mu = torch.clamp(mu, min=-5.0, max=5.0)  # 범위 완화
        logvar = torch.clamp(logvar, min=-4.0, max=5.0)

        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            logger.warning("NaN in mu/logvar at encoder output")

        return mu, logvar

    def kl_loss(self):
        return self.fc1.kl_loss()

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, use_skip=True):
        super().__init__()
        self.current_epoch = 0
        self.use_skip = use_skip
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(latent_dim, 256)
        self.dropout = nn.Dropout(0.3)

        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )

        self.bi_lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 256, kernel_size=5, padding=2)
        )

        if use_skip:
            self.skip_proj = nn.Linear(latent_dim, 256)
            self.alpha = nn.Parameter(torch.tensor(0.05))

        self.postnet = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def forward(self, z):
        # [선택적] 다양성 확보용 noise 주입은 epoch에 따라 적용할 수 있도록 조정
        if self.training and self.current_epoch <= 10:
            z += 0.03 * torch.randn_like(z)

        z = F.dropout(z, p=0.2, training=self.training)

        x = self.fc(z)
        x = self.dropout(x).unsqueeze(1)

        lstm_out, _ = self.bi_lstm(x)
        conv_input = lstm_out.transpose(1, 2)
        base_out = self.conv_layers(conv_input)

        if base_out.shape[-1] != self.output_dim:
            base_out = F.interpolate(base_out, size=self.output_dim, mode="linear", align_corners=False)

        z_context = self.z_proj(z).unsqueeze(2)
        base_out = base_out + z_context

        if self.use_skip:
            skip_out = self.skip_proj(z).unsqueeze(2)
            skip_out = skip_out.expand(-1, -1, base_out.shape[-1])
            alpha = torch.sigmoid(self.alpha)
            out = alpha * base_out + (1 - alpha) * skip_out
        else:
            out = base_out

        refined = self.postnet(out)
        out = out + refined  # residual connection 추가

        out = out.squeeze(1) if out.shape[1] == 1 else out.mean(dim=1)

        return out

    def kl_loss(self):
        return 0.0


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
        self.encoder = Encoder(latent_dim, dist_type, df)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = sample_latent(mu, logvar, self.dist_type, self.df)
        z = F.dropout(z, p=0.1, training=self.training)

        if torch.isnan(z).any() or z.std() > 10.0:
            logger.warning(f"[z warning] z has extreme std: {z.std().item():.4f}")

        if mu.std() < 0.01 or logvar.std() < 0.01:
            logger.warning(f"[Collapse] mu.std={mu.std():.4f}, logvar.std={logvar.std():.4f}")

        recon_x = self.decoder(z)

        if self.training and recon_x.std() < 0.01:
            recon_x += 0.01 * torch.randn_like(recon_x)
        if torch.isnan(z).any():
            logger.error("NaN detected in latent z!")
        if torch.isnan(mu).any():
            logger.error("NaN detected in mu!")
        if torch.isnan(logvar).any():
            logger.error("NaN detected in logvar!")

        return recon_x, mu, logvar, z

    def kl_loss(self):
        return self.encoder.kl_loss() + self.decoder.kl_loss()