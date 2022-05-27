import numpy as np
import librosa

from data_augmentation import DataAugmentation as Augment


class MFCC:
    def __init__(self):
        # sample_rate = 22050
        self.sample_rate = 22050


    def extract_features(self, data):

        mfccs = librosa.feature.mfcc(y=data, sr=self.sample_rate, n_mfcc=58)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        result = np.array(mfccs_processed)

        return result

    def get_features(self, path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast')

        # without augmentation
        res1 = self.extract_features(data)
        result = np.array(res1)

        # noised
        noise_data = Augment().noise(data)
        res2 = self.extract_features(noise_data)
        result = np.vstack((result, res2))  # stacking vertically

        # stretched
        stretch_data = Augment().stretch(data)
        res3 = self.extract_features(stretch_data)
        result = np.vstack((result, res3))

        # shifted
        shift_data = Augment().shift(data)
        res4 = self.extract_features(shift_data)
        result = np.vstack((result, res4))

        # pitched
        pitch_data = Augment().pitch(data, sample_rate)
        res5 = self.extract_features(pitch_data)
        result = np.vstack((result, res5))

        # speed up
        higher_speed_data = Augment().stretch(data, rate=0.75)
        res6 = self.extract_features(higher_speed_data)
        result = np.vstack((result, res6))

        # speed down
        lower_speed_data = Augment().stretch(data, rate=1.25)
        res7 = self.extract_features(lower_speed_data)
        result = np.vstack((result, res7))

        return result

