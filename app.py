import os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import json
import soundfile
import faiss
from settings import DURATION, WAVE_OUTPUT_FILE, ENDPOINT_NAME
from src.sound import sound
from sagemaker.predictor import Predictor
from sagemaker.s3 import S3Downloader
from sagemaker.session import Session


logger = logging.getLogger('app')


def download_data_from_s3(sagemaker_session):
    S3Downloader.download(
        s3_uri='s3://wav2vec2-music-recognition/songs_db_filenames.npy',
        local_path='.',
        sagemaker_session=sagemaker_session
    )
    S3Downloader.download(
        s3_uri='s3://wav2vec2-music-recognition/songs_db.index',
        local_path='.',
        sagemaker_session=sagemaker_session,
    )


def inference_handler(predictor, index, filenames):
    print("inferenicing and calculating closest songs...")
    audio_array, sampling_rate = soundfile.read(WAVE_OUTPUT_FILE)
    json_request_data = {
        "speech_array": audio_array.tolist(),
        "sampling_rate": sampling_rate
        }
    
    with st.spinner("Classifying the chord"):
        prediction = predictor.predict(json.dumps(json_request_data).encode('utf-8'))
    
    x = json.loads(prediction.decode('utf-8'))
    xq = np.array([json.loads(x[0])])

    k = 3
    _, k_index = index.search(xq, k) 
    return [filenames[index] for index in k_index[0]]


def main():
    sagemaker_session = Session()
    os.environ['AWS_PROFILE'] = "default"
    os.environ['AWS_DEFAULT_REGION'] = 'ap-southeast-1'
    endpoint_name = "huggingface-wav2vec2-endpoint-1709692846"


    download_data_from_s3(sagemaker_session)
    predictor = Predictor(endpoint_name=endpoint_name)
    index = faiss.read_index("songs_db.index")
    filenames = np.load("songs_db_filenames.npy")


    title = "Music Recognition Demo"
    st.title(title)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        top_k_songs = inference_handler(predictor, index, filenames)
        st.success("Classification completed")
        st.write("### These are top possible songs ###")
        for name in top_k_songs:
            st.markdown("- " + name)


if __name__ == '__main__':
    main()