import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.signal import resample
import math
from scipy import signal
import plotly.graph_objects as go
import os
from skimage.transform import resize
from PIL import Image
import shutil

np.arange(3)

# Load dataset
file_path = './image_generation/data/dataset_training_aug.h5'
hdf = h5py.File(file_path, 'r')

list(hdf.keys())
hdf.attrs.keys()
data = hdf['data']
label = hdf['label']
cfo = hdf['CFO']
rss = hdf['RSS']
print(f"Data group shape: {data.shape}")
print(f"Label group shape: {label.shape}")
print(f"CFO group shape: {cfo.shape}")
print(f"RSS group shape: {rss.shape}")
print(f"Data group dtype: {data.dtype}")
print(f"Label group dtype: {label.dtype}")
print(f"CFO group dtype: {cfo.dtype}")
print(f"RSS group dtype: {rss.dtype}")

##load complex data
print("data shapes :")
data_1 = data[:999, :]
print(data_1.shape)
I_1 = np.reshape(data_1[:, :8192], 8192 * 999) # Reshape 1d array
Q_1 = np.reshape(data_1[:, 8192:16384], 8192 * 999)
complex_signal = I_1 + 1j*Q_1
print("I1 : "+str(I_1.shape))
print("Q1 : "+str(Q_1.shape))
print("complex_signal : "+str(complex_signal.shape))

# Spectrogram
freq, time, spectrogram = signal.stft(complex_signal,window='boxcar',nperseg=256,noverlap=128,nfft=256,return_onesided=False,padded=False,boundary=None)

print(freq.shape)
print(time.shape)
print(spectrogram.shape)

spectrogram = np.fft.fftshift(spectrogram, axes=0)
spectrogram_amplitude = np.log10(np.abs(spectrogram) ** 2)

# plt.figure(figsize=(8, 4))
# plt.title(f'Time-Frequency Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.imshow(spectrogram_amplitude[:, :64], cmap='viridis', aspect='auto', origin='lower', norm='linear')
# plt.colorbar(label='Log Amplitude')


short_complex_signal = complex_signal[:8192]
N = len(short_complex_signal)

# Finding the right alpha
taus = np.arange(0, 600)
alphas = np.arange(-0.3, 0.3, 0.005)
CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
for j in range(len(alphas)):
    for i in range(len(taus)):
        CAF[j, i] = np.sum(short_complex_signal *
                    np.conj(np.roll(short_complex_signal, taus[i])) *
                    np.exp(-2j * np.pi * alphas[j] * np.arange(N)))


CAF_magnitudes = np.average(np.abs(CAF), axis=1) # at each alpha, calc power in the CAF
# plt.plot(alphas, CAF_magnitudes)
# plt.xlabel('Alpha')
# plt.ylabel('CAF Power')


#suite
np.argmax(CAF_magnitudes)
CAF2=CAF.copy()
CAF_magnitudes = np.average(np.abs(CAF2), axis=1) # at each alpha, calc power in the CAF
# plt.plot(alphas, CAF_magnitudes)
# plt.xlabel('Alpha')
# plt.ylabel('CAF Power')

extent = (0, 600, float(np.max(alphas)), float(np.min(alphas)))
plt.imshow(np.abs(CAF2), extent=extent, aspect='auto', vmax=np.max(np.abs(CAF2))/2)
plt.colorbar(label='CAF (tau, alpha)')
plt.title('CAF')
plt.xlabel('taus')
plt.ylabel('alphas')
plt.show()
