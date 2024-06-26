import torchaudio
import torch

import librosa


def load_audio(filename, normalize=True, channel=0):
    waveform, sample_rate = torchaudio.load(filename, normalize=normalize)
    num_channels, num_frames = waveform.shape
    if num_channels != 1:
        x = librosa.to_mono(waveform.numpy())
        waveform = torch.tensor(x).float()
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate