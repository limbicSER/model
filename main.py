import asyncio
import io
import sys

import keras.models
import librosa
import numpy as np
import soundfile as sf
from grpc import aio
from sklearn.preprocessing import OneHotEncoder

from Protos.Generated.speech_emotion_recognition_pb2 import LoadDataReply
from Protos.Generated.speech_emotion_recognition_pb2_grpc import SpeechEmotionRecognitionServicer, \
    add_SpeechEmotionRecognitionServicer_to_server
from dataset_preparation import Datatset
from feature_extraction import MFCC

model = keras.models.load_model("saved_models/total_model.h5")
model.summary()
mfcc = MFCC()

encoder = OneHotEncoder()
encoder.fit_transform(np.array(['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']).reshape(-1, 1)).toarray()
y_map = {
    "angry": 0,
    "calm": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "neutral": 5,
    "sad": 6,
    "surprise": 7
}


class SpeechEmotionRecognitionService(SpeechEmotionRecognitionServicer):
    async def LoadData(self, request_iterator, context):
        request_bytes = b''
        async for req in request_iterator:
            request_bytes += req.Content

        with open("grpc_audios/e1.wav","wb+") as f:
            f.write(request_bytes)

        print("Try read")
        # data, sampling_rate = sf.read(io.BytesIO(request_bytes))
        data, sampling_rate = librosa.core.load(io.BytesIO(request_bytes))
        sample = mfcc.extract_features(data)
        print(sample)
        sample = sample.reshape(1, 58)

        probas = model.predict(sample)
        probas_percent = [round(p * 100, 2) for p in probas[0]]
        # print(np.array(males.labels).reshape(-1, 1))
        y_pred = encoder.inverse_transform(probas)
        print(probas_percent)
        # print(, max(probas_percent))
        # print(f"{round(time.time() - start, 3)} seconds")
        print(y_pred)
        reply: LoadDataReply = LoadDataReply(emotion=y_map[y_pred[0][0]])
        return reply


async def serve():
    server = aio.server()
    add_SpeechEmotionRecognitionServicer_to_server(SpeechEmotionRecognitionService(), server)

    listen_addr = "[::]:50052"
    server.add_insecure_port(listen_addr)
    print(f"Starting server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
        policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)

    asyncio.run(serve())


