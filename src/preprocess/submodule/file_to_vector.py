# TODO : make parameter global

import numpy as np
import librosa
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(SRC_DIR)

from utils.submodule.load_audio import load_audio


def file_to_vector_mel(file_name,
                         n_mels=64,
                         frames=2,
                         n_fft=2048,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    '''
    file_name 으로 log mel spectrogram을 만들어주는 함수

    input : 
        file_name : str, target .wav
        n_mels : int, number of mel bins
        frames : int, number of frames to concatenate
        n_fft : int, length of the FFT window
        hop_length : int, number of samples between successive frames
        power : float, power to apply to the mel spectrogram

    output :
        log_mel_spectrogram : np.array, log mel spectrogram
    '''
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    y, sr = load_audio(file_name)
    y = y[0].numpy()
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # output datatype is np.float32

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    # vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1


    # # 06 generate feature vectors by concatenating multi_frames
    # vectorarray = np.zeros((vectorarray_size, dims), float)
    # for t in range(frames):
    #     vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return log_mel_spectrogram


def file_to_vector_chroma(file_name,
                         n_chroma=12,
                         d=5,
                         win_len_smooth=41,
                         frames=1, #not in use
                         hop_length=512):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    '''
    file_name 으로 chroma cens를 만들어주는 함수

    input : 
        file_name : str, target .wav
        n_chroma : int, number of chroma bins
        d : int, interval for column extraction
        win_len_smooth : int, length of the smoothing window
        frames : int, number of frames to concatenate
        hop_length : int, number of samples between successive frames

    output :
        cens_gram : np.array, chroma cens
    '''
    # 01 calculate the number of dimensions
    dims = n_chroma * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    y, sr = load_audio(file_name)
    y = y[0].numpy()
    cens_gram = librosa.feature.chroma_cens(y=y,
                                            sr=sr,
                                            n_chroma=n_chroma,
                                            win_len_smooth=win_len_smooth,
                                            bins_per_octave=n_chroma*3)

    return cens_gram[:, ::d]