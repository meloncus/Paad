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


def get_df_feat(df, n_fft, sr, means=False):
    ''' Used to extract Features from spectrograms 
    MFCC, Log mel energy and Chroma (CENS)
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


def get_features(file, n_fft, sr, frac=10, d=5, win_len_smooth=41, means=False):
    n_fft = n_fft
    x, sr = load_audio(file)
    
    lmfe, lml = get_lmfe(x, n_fft, sr, means=means)

    mfcc, mfl = get_mfcc(x, n_fft, sr, frac=10, means=means)

    chroma, cl = get_chroma(x, sr, d=d, win_len_smooth=win_len_smooth, means=means)

    feat = np.concatenate(([lmfe], [mfcc], [chroma]), axis=1).flatten()
    labels = np.concatenate(([lml], [mfl], [cl]), axis=1)
    return feat.flatten(), labels

def get_lmfe(y, n_fft, sr, means=False):
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


def get_mfcc(x, n_fft, sr, frac=10, means=False):
    ''' MFCC - effective in capturing spectral features that are relevant to human perception
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



                     
def get_chroma(x, sr, d=5, win_len_smooth=41, means=False):
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
    

def convert_complex_to_real(value):
    return complex(value).real