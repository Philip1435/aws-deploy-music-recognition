import soundfile
import sagemaker
import boto3
import os 


# sagemaker = boto3.client('sagemaker-runtime', 'ap-southeast-1')
speech_array, sampling_rate = soundfile.read('output/recording/recorded.wav')

BUCKET="wav2vec2-music-recognition" # please use your bucket name
PREFIX = "huggingface-blog" 


# session = sagemaker.Session()
# s3 = boto3.client('s3')
sm = boto3.Session(region_name='ap-southeast-1')
from sagemaker.predictor import Predictor
print('adsfasdfasf')

predictor = Predictor(endpoint_name='huggingface-wav2vec2-endpoint-1709692846')

json_request_data = {"speech_array": speech_array.tolist(),
                     "sampling_rate": sampling_rate}

prediction = predictor.predict(json_request_data)
print(prediction)
print('success')