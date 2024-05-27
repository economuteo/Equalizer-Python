from pydub import AudioSegment
import numpy as np

# Load the mp3 file
audio = AudioSegment.from_mp3("Dr. Dre - Still (Instrumental).mp3")

# Convert to mono
audio = audio.set_channels(1)

# Get the raw audio data
raw_data = audio.get_array_of_samples()

# Convert to a numpy array
np_array = np.array(raw_data)