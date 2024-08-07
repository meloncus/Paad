import numpy as np
import pandas as pd
import torchaudio
import speechpy
import librosa
import torch
import os
import sys

from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(SRC_DIR)

from utils.submodule.load_audio import load_audio


def get_df_feat (df, n_fft, sr, means=False) :
    ''' Used to extract Features from spectrograms 
    MFCC, Log mel energy and Chroma (CENS)
    '''
    '''
    dataset 에서 불러온 데이터를 전처리하기 위한 함수
    MFCC, Log mel energy, Chroma (CENS)를 추출하는데 사용됨

    input : 
        df : pandas.DataFrame, dataframe
        n_fft : int, length of the FFT window
        sr : int, sample rate
        means : bool, whether to return the mean of the features

    output :
        df : pandas.DataFrame, dataframe
    '''
    feat_cols = []

    # Initialize the progress bar
    progress_bar = tqdm(total=len(df), position=0, leave=True)
    for i, row in df.iterrows():
        filename = row['filename']
        feat, labels = get_features(filename, n_fft, sr, frac=10, means=means)

        feat_cols.append(feat)
        lab_cols = labels
        # Update the progress bar
        progress_bar.update(1)

    feat_array = np.vstack(feat_cols)
    lab_array = lab_cols.flatten()

    feat_df = pd.DataFrame(feat_array, columns=list(lab_array), index=df.index)

    # Convert complex numbers to real values
    feat_df = feat_df.applymap(convert_complex_to_real)

    # Assign the columns to the original DataFrame
    df = pd.concat([df, feat_df], axis=1)

    return df


def get_features (file, n_fft, sr, frac=10, d=5, win_len_smooth=41, means=False) :
    '''
    audio file에서 mel, chroma feature를 추출하는 함수

    input :
        file : str, audio file
        n_fft : int, length of the FFT window
        sr : int, sample rate
        frac : int, fraction of the window length
        d : int, interval to extract columns
        win_len_smooth : int, window length for smoothing
        means : bool, whether to return the mean of the features

    output :
        feat.flatten() : np.array, features
            feat : np.array, features
        labels : list, labels
    '''
    n_fft = n_fft
    x, sr = load_audio(file)
    
    lmfe, lml = get_lmfe(x, n_fft, sr, means=means)

    mfcc, mfl = get_mfcc(x, n_fft, sr, frac=10, means=means)

    chroma, cl = get_chroma(x, sr, d=d, win_len_smooth=win_len_smooth, means=means)

    feat = np.concatenate(([lmfe], [mfcc], [chroma]), axis=1).flatten()
    labels = np.concatenate(([lml], [mfl], [cl]), axis=1)
    return feat.flatten(), labels

def get_lmfe (y, n_fft, sr, means=False) :
    '''
    Log mel energy - captures the energy of the signal in different frequency bands
    Log mel energy - 신호의 에너지를 다른 주파수 대역에서 캡처

    input :
        y : torch.Tensor, audio signal
        n_fft : int, length of the FFT window
        sr : int, sample rate
        means : bool, whether to return the mean of the features

    output :
        lmfe.flatten() : np.array, log mel energy
            lmfe : np.array, log mel energy
        labels : list, labels
    '''
    n_fft=n_fft
    y = y[0].numpy()
    n=64

    # Compute log mel energy
    lmfe = speechpy.feature.lmfe(y, sr, fft_length=n_fft, frame_length=1, frame_stride=(0.75), num_filters=n)
    labels = []
    if not means:
        for i in range(1, n+1):
            for j in range(lmfe.shape[0]):
                labels.append('lmfe{}_{}'.format(i+1, j+1))
        return lmfe.flatten(), labels
    if means:
        lmfe = np.mean(lmfe, axis=0)
        
        for j in range(lmfe.shape[0]):
            labels.append('lmfe{}'.format(j+1))
            
        return lmfe.flatten(), labels


def get_mfcc (x, n_fft, sr, frac=10, means=False) :
    ''' MFCC - effective in capturing spectral features that are relevant to human perception
    '''
    '''
    MFCC - 인간의 지각에 관련된 스펙트럼 특징을 효과적으로 캡처

    input :
        x : torch.Tensor, audio signal
        n_fft : int, length of the FFT window
        sr : int, sample rate
        frac : int, fraction of the window length
        means : bool, whether to return the mean of the features

    output :
        mfcc.flatten() : np.array, mfcc
            mfcc : np.array, mfcc
        labels : list, labels
    '''
    n_fft=sr

    win_length = sr // frac # (0.1 seconds)
    hop_length = int(win_length*0.75) # 25% overlap 

    n_mfcc = 13
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr,n_mfcc=n_mfcc, log_mels=True,
                                                melkwargs={
                                                    "n_fft": n_fft,
                                                    "win_length": win_length,
                                                    "hop_length": hop_length,
                                                    "n_mels": 64,
                                                    "mel_scale": "htk",
                                                },)

    mfcc = mfcc_transform(x)
    mfcc = mfcc[:, 1:, :]  # Drop the first MFCC coefficient
    
    labels = []
    if not means:
        for i in range(1, n_mfcc):
            for j in range(mfcc.shape[2]):
                labels.append('mfcc{}_{}'.format(i+1, j+1))
                
        return np.squeeze(mfcc.numpy()).flatten(), labels
    if means:
        mfcc = torch.mean(mfcc, 2)
        for i in range(mfcc.shape[1]):
             labels.append('mfcc{}'.format(i+1))
        return mfcc.numpy().flatten(), labels



                     
def get_chroma (x, sr, d=5, win_len_smooth=41, means=False) :
    '''
    Chroma - captures the harmonic content of the signal
    Chroma - 신호의 음향 내용을 캡처, 음악에서 음의 높낮이를 나타냄

    input :
        x : torch.Tensor, audio signal
        sr : int, sample rate
        d : int, interval to extract columns
        win_len_smooth : int, window length for smoothing
        means : bool, whether to return the mean of the features

    output :
        chroma.flatten() : np.array, chroma
            chroma : np.array, chroma
        labels : list, labels
    '''
    hop_length = 512
    chroma = librosa.feature.chroma_cens(y=x[0].numpy(), sr=sr, win_len_smooth=win_len_smooth, hop_length=hop_length)
    chroma = chroma[:, ::d]  # d 간격으로 열 추출

    labels = []
    if not means:
        # 인덱스 검사 추가 # 인덱스를 1, 2로 잡아놨었음 0, 1로 잡아야하는데
        if chroma.shape[0] > 0 and chroma.shape[1] > 0:
            for i in range(chroma.shape[0]):
                for j in range(chroma.shape[1]):
                    labels.append('chroma{}_{}'.format(i+1, j+1))
        else:
            # 큰 데이터가 없는 경우 처리 (예: 에러 발생, 빈 레이블 반환)
            pass  # 추후 필요에 따라 로직 추가

        return np.squeeze(chroma).flatten(), labels
    if means:
        chroma = np.mean(chroma, axis=1)
        for i in range(chroma.shape[0]):
             labels.append('chroma{}'.format(i+1))
        return chroma, labels
    

def convert_complex_to_real (value) :
    '''
    복소수를 실수로 변환하는 함수

    input :
        value : complex, complex number

    output :
        complex(value).real : float, real number
    '''
    return complex(value).real