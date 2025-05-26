# Bayesian VAE for Speech Enhancement
This project implements a Bayesian Variational Autoencoder (Bayesian VAE) model for speech enhancement, particularly targeting the VoiceBank-DEMAND dataset.

## 🚀 Usage
### Model pre-training
```
python main.py \
  --train \
  --clean_dir /home/adl/workspace/denoise/VoiceBank-DEMAND/clean_trainset_56spk_wav \
  --noisy_dir /home/adl/workspace/denoise/VoiceBank-DEMAND/noisy_trainset_56spk_wav \
  --train_val_ratio 0.8 \
  --epoch 30 \
  --patience 32 \
  --batch 32 \
  --pdf gaussian \
  --latent_dim 128 \
  --report
```

### Model fine-tuning
```
python main.py \
  --train \
  --ft \
  --clean_dir /home/adl/workspace/denoise/fine-tuning/clean_trainset_10spk_wav \
  --noisy_dir /home/adl/workspace/denoise/fine-tuning/noisy_trainset_10spk_wav \
  --train_val_ratio 0.8 \
  --epoch 30 \
  --patience 32 \
  --batch 32 \
  --pdf gaussian \
  --latent_dim 128 \
  --report
```

### Model test
```
python main.py \
--test
--output_dir /home/adl/workspace/denoise/output \
--manifest /home/adl/workspace/denoise/output/bvae_20250516T182506.manifest \
--test_input /home/adl/workspace/denoise/VoiceBank-DEMAND/noisy_testset_wav/p257_434.wav \
```

## ⚙️ Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--train` | Enables training mode. Without this flag, training will not run. |
| `--ft` | Enables fine-tuning mode (typically with a smaller dataset). |
| `--test` | Runs the model in test/inference mode. |
| `--clean_dir` | Path to the directory containing clean (target) speech files. |
| `--noisy_dir` | Path to the directory containing noisy (input) speech files. |
| `--train_val_ratio` | Ratio to split dataset into training and validation sets (e.g., `0.8` means 80% training, 20% validation). |
| `--epoch` | Number of total training epochs. |
| `--patience` | Patience setting for early stopping. Training stops if validation loss does not improve for this many epochs. |
| `--batch` | Batch size for training. |
| `--pdf` | Probability distribution function for the latent variable. Options: `gaussian`, `student-t`. |
| `--latent_dim` | Dimensionality of the latent space. |
| `--report` | Enables training report generation (e.g., loss curves, summary stats). Outputs to `report/` directory. |
| `--output_dir` | Output directory where the enhanced audio and manifest will be saved during test. |
| `--manifest` | Path to the trained model manifest file. |
| `--test_input` | Input noisy WAV file for testing (single file). |

## 🚀 설치 방법
### CPU-Only
```
pip install torch torchvision torchaudio
pip install torchmetrics
pip install pystoi pesq
pip install librosa
pip install --upgrade torchmetrics
```

### CUDA 12.1
```
pip install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8
```
pip install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu118
```

## 📂 Public Dataset for speech 
- [VoiceBank-DEMAND] (https://datashare.ed.ac.uk/handle/10283/2791)

## 📄 License
MIT License © 2025 Hakjae Kim (krunivs@gmail.com)
