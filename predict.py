import io
import os
import time

import keras.models
import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from dataset_preparation import Datatset
from pydub import AudioSegment
import soundfile as sf
from feature_extraction import MFCC

model = keras.models.load_model("saved_models/total_model.h5")
model.summary()
mfcc = MFCC()

dp = Datatset()
_, males = dp.create_dfs()

encoder = OneHotEncoder()
encoder.fit_transform(np.array(males.labels).reshape(-1, 1)).toarray()

while True:
    try:
        audio_path: str = input("Audio file path: ")
        if audio_path == "stop":
            break
        start = time.time()
        # if audio_path.split('.')[-1] != 'wav':
        #     old = audio_path
        #     # https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
        #     input_audio = AudioSegment.from_file(audio_path)
        #     audio_path = f"{''.join(audio_path.split('.')[:-1])}.wav"
        #     input_audio.export(audio_path, format='wav')
        #     os.remove(old)

        # data, sampling_rate = librosa.load(audio_path)
        with open("input_audios/e6.wav", 'rb') as f:
            audio_bytes = f.read()

        data, sampling_rate = librosa.core.load(io.BytesIO(audio_bytes))
        sample = mfcc.extract_features(data)
        print(sample)
        sample = sample.reshape(1, 58)

        probas = model.predict(sample)
        probas_percent = [round(p * 100, 2) for p in probas[0]]
        y_pred = encoder.inverse_transform(probas)
        print(probas_percent)
        print(y_pred, max(probas_percent))
        print(f"{round(time.time() - start, 3)} seconds")
    except:
        continue
