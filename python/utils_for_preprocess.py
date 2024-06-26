import torchaudio
import torch

import librosa


def load_audio(filename, normalize=True, channel=0) :
    '''
    load audio file and return waveform and sample rate

    input
    filename : str
        target .wav file
    normalize : bool (default : True)
    channel : int (default : 0)

    output
    waveform : torch.tensor
    sample_rate : int
    '''

    waveform, sample_rate = torchaudio.load(filename, normalize=normalize)

    # if waveform is stereo, number of channels is 2
    num_channels, num_frames = waveform.shape
    if num_channels != 1:
        x = librosa.to_mono(waveform.numpy())
        waveform = torch.tensor(x).float() # convert to torch.tensor and float
        waveform = waveform.unsqueeze(0) # add channel dimension [1, num_frames]
    return waveform, sample_rate