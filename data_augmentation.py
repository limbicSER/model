import numpy as np
import librosa


class DataAugmentation:
    def __init__(self):
        pass

    def noise(self, data):
        noise_amp = 0.04 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data, rate=0.70):
        return librosa.effects.time_stretch(data, rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.8):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
