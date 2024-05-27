from pydub import AudioSegment
from scipy.signal import iirfilter, lfilter
from scipy.signal import butter
from scipy.signal import cheby1
from scipy.signal import ellip
from scipy.signal import bessel
import numpy as np
import matplotlib.pyplot as plt

# Load the mp3 file
audio = AudioSegment.from_mp3("./InputData/Still.mp3")
# Convert to mono
audio = audio.set_channels(1)
# Get the raw audio data
raw_data = audio.get_array_of_samples()
# Convert to a numpy array
np_array = np.array(raw_data)


# Plot the frequency response of the audio

def plot_frequency_response(data, fs, title):
    n = len(data)
    T = 1.0 / fs
    yf = np.fft.fft(data, n=4*n)  # Increase FFT resolution by zero-padding
    xf = np.linspace(0.0, 1.0/(2.0*T), 2*n)
    plt.plot(xf, 2.0/n * np.abs(yf[0:2*n]))
    plt.grid()
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()


# Filters
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def cheby_lowpass(cutoff, fs, ripple_db, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, ripple_db, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(data, cutoff, fs, ripple_db, order=5):
    b, a = cheby_lowpass(cutoff, fs, ripple_db, order=order)
    y = lfilter(b, a, data)
    return y


def ellip_lowpass(cutoff, fs, ripple_db, attenuation_db, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = ellip(order, ripple_db, attenuation_db,
                 normal_cutoff, btype='low', analog=False)
    return b, a


def ellip_lowpass_filter(data, cutoff, fs, ripple_db, attenuation_db, order=5):
    b, a = ellip_lowpass(cutoff, fs, ripple_db, attenuation_db, order=order)
    y = lfilter(b, a, data)
    return y


def bessel_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = bessel(order, normal_cutoff, btype='low', analog=False)
    return b, a


def bessel_lowpass_filter(data, cutoff, fs, order=5):
    b, a = bessel_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Plot frequency response before applying the filter
plot_frequency_response(np_array, 44100, 'Before applying the filter')

# Apply the filter with negative gain
filtered_data = butter_lowpass_filter(
    np_array, 2000, 44100, order=5)

# Plot frequency response after applying the filter
plot_frequency_response(filtered_data, 44100, 'After applying the filter')

# Convert filtered data to int16 to avoid overflow when creating the AudioSegment
filtered_data_int16 = np.int16(filtered_data)

# Create a new AudioSegment from the filtered data
filtered_audio = AudioSegment(
    filtered_data_int16.tobytes(),
    frame_rate=44100,
    sample_width=filtered_data_int16.dtype.itemsize,
    channels=1
)

# Download the filtered audio
filtered_audio.export("./OutputData/filtered_audio.mp3", format="mp3")
