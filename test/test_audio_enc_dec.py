"""
audio wav encode and decode test
"""
import torch
import torchaudio
import torch.nn.functional as F

from denoise_bayes_vae.bayes_vae import BayesianVAE


def main():
    target_sr = 16000
    input_dim = 16000
    input_file_path = '/home/adl/workspace/denoise/VoiceBank-DEMAND/noisy_testset_wav/p257_434.wav'
    output_file_path = '/home/adl/workspace/denoise/output/test2.wav'
    waveform, input_sr = torchaudio.load(input_file_path) # shape: [1, L] or [2, L]

    channels = waveform.shape[0]    # 1: mono, 2: stereo
    samples = waveform.shape[1]     # samples
    print(f'channel = {channels}, samples = {samples}')

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resampling to 16KHz
    if input_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Flatten to [L]
    waveform = waveform.squeeze(0)

    # Slice into input_dim-length chunks (e.g., 16000)
    segments = []
    denoised = []
    total_samples = waveform.shape[0]

    for start in range(0, total_samples, input_dim):
        end = start + input_dim
        chunk = waveform[start:end]

        if chunk.shape[0] < input_dim:
            chunk = F.pad(chunk, (0, input_dim - chunk.shape[0]))

        segments.append(chunk)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file = "/home/adl/workspace/denoise/output/bvae_20250516T182506.pth"
    model = BayesianVAE(input_dim, 128, "gaussian", 3.0)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for chunk in segments:
            input_tensor = chunk.to(device).unsqueeze(0)  # [1, 16000]
            recon, _, _ = model(input_tensor)
            recon = recon.squeeze(0).cpu()
            recon = recon / (recon.abs().max() + 1e-9)  # Output normalization
            denoised.append(recon)

    # Concatenate all chunks
    segments = denoised
    waveform = torch.cat(segments, dim=0)

    # Resample back to original SR if needed
    if input_sr != target_sr:
        resample = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=input_sr)
        waveform = resample(waveform)

    waveform = waveform.unsqueeze(0)

    # save to file
    torchaudio.save(output_file_path, waveform, input_sr)

if __name__ == "__main__":
    main()