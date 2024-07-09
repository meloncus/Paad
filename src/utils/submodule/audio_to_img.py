import numpy as np
import matplotlib.pyplot as plt

DURATION = 10
SAMPLING_RATE = 16000

def plot_waveform (waveform, duration = DURATION, sampling_rate = SAMPLING_RATE) :
    '''
    mono data only plot
    '''
    time = np.linspace(0., duration, sampling_rate * duration)
    waveform = np.squeeze(waveform)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform)
    plt.title("Audio Waveform in Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()