# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 18:34:40 2025

@author: ksrik
"""

#%% Load Libraries

import os
import librosa
import numpy as np
from scipy.signal import welch, periodogram, butter, filtfilt
import pywt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

#%% Load File

folder = r"E:\Personal\Projects\Audio Processing\Data"
filename = r"Aha Naa Pelliyanta.mp3"

file = os.path.join(folder, filename)
signal, sr = librosa.load(file, sr=None)
time = np.linspace(0, len(signal)/sr, len(signal))

duration = 5*sr
start = 0
end = start + duration
# end = len(signal)

signal = signal[start:end]
time = time[start:end]

f, power = periodogram(signal, fs = sr)

#%% Filter
N = 2
Wn = [50, 80]

b, a = butter(N, Wn, 'bandpass', fs = sr)
filtered = filtfilt(b, a,  signal)

f, power_filt = periodogram(filtered, fs = sr)

print("NaNs in signal:", np.any(np.isnan(signal)))
print("Infs in signal:", np.any(np.isinf(signal)))
print("Signal min/max:", np.min(signal), np.max(signal))


#%% Plot

fig = make_subplots(2, 1)
fig.add_trace(go.Scatter(x = time, y = signal, name = "Audio"), row=1, col = 1)
fig.add_trace(go.Scatter(x = time, y = filtered, name = f"Filtered = {Wn}"), row=1, col = 1)

fig.add_trace(go.Scatter(x = f, y = power, name = "Frequency"), row=2, col = 1)
fig.add_trace(go.Scatter(x = f, y = power_filt, name = f"Filtered Frequency = {Wn}"), row=2, col = 1)
fig.show()
