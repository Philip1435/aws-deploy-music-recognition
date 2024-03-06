import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


OUT_DIR = os.path.join(ROOT_DIR, 'output/')
RECORDING_DIR = os.path.join(OUT_DIR, 'recording')

WAVE_OUTPUT_FILE = os.path.join(RECORDING_DIR, "recorded.wav")
SPECTROGRAM_FILE = os.path.join(RECORDING_DIR, "spectrogram.png")

ENDPOINT_NAME = 'huggingface-pytorch-inference-2024-03-06-00-27-29-997'


# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 16000   # Default sample rate of microphone or recording device
DURATION = 4   # 3 seconds
CHUNK_SIZE = 1024