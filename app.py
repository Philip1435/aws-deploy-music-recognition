import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from settings import DURATION, WAVE_OUTPUT_FILE, ENDPOINT_NAME
from src.sound import sound
import soundfile
from sagemaker.predictor import Predictor
import boto3
import json
import sagemaker

logger = logging.getLogger('app')


def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.pyplot(clear_figure=False)

def main():

    BUCKET="wav2vec2-music-recognition" # please use your bucket name

    title = "Music Recognition Demo"
    st.title(title)
    runtime = boto3.client('sagemaker-runtime', 'ap-southeast-1')


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
        audio_array, sampling_rate = soundfile.read(WAVE_OUTPUT_FILE)
        json_request_data = {
            "speech_array": audio_array.tolist(),
            "sampling_rate": sampling_rate
            }
        
        with st.spinner("Classifying the chord"):
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(json_request_data).encode('utf-8')
            )
            # prediction = predictor.predict(json.dumps(json_request_data).encode('utf-8'))
            # xq = json.loads(response['Body'])
        
        res = response['Body'].read().decode('utf-8')

        
        st.success("Classification completed")

        st.write("### The recorded chord is **" + "**")

        st.write("\n")

if __name__ == '__main__':
    main()
    # for i in range(100):
    #   # Update the progress bar with each iteration.
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.1)