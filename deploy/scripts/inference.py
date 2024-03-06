import logging
import json
import torch

from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_path = '/opt/ml/model'
logger.info("Libraries are loaded")


def model_fn(model_dir):
    device = get_device()
    model = Wav2Vec2Model.from_pretrained(model_path).to(device)
    # model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    logger.info("Model is loaded")

    return model


def input_fn(json_request_data, content_type='application/json'):

    input_data = json.loads(json_request_data)
    logger.info("Input data is processed")

    return input_data


def predict_fn(input_data, model):

    logger.info("Starting inference.")
    device = get_device()

    logger.info(input_data)

    speech_array = input_data['speech_array']
    sampling_rate = input_data['sampling_rate']

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    input_values = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(device)

    with torch.no_grad():
        o = model(input_values)
    return o.last_hidden_state.squeeze(dim=0).mean(dim=0).numpy().tolist()


def output_fn(transcript, accept='application/json'):
    logger.info(0000000000)
    return json.dumps(transcript), accept

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device