import streamlit as st
import io
import librosa
from scipy.io import wavfile
from pydub import AudioSegment
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)
    return virtualfile

def plot_wav(wav, max_audio_len):
    padding = np.zeros(max_audio_len - wav.shape[0])
    fig = plt.figure()
    if wav.ndim == 1: # mono
        ax = fig.add_subplot(211)
        ax.plot(np.concatenate([wav, padding]))
    elif wav.ndim == 2: # stereo
        ax1 = fig.add_subplot(211, xlabel="L")
        ax2 = fig.add_subplot(212, xlabel="R")
        ax1.plot(np.concatenate(wav[:, 0], padding))
        ax2.plot(np.concatenate(wav[:, 1], padding))
        fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    return st.pyplot(fig)

st.title("Sound Mixer")

uploaded_files = st.file_uploader("Upload your audio file (.wav)", type=["wav"], accept_multiple_files=True)

is_file_uploaded = uploaded_files is None or len(uploaded_files) != 0
if not is_file_uploaded:
    uploaded_files = ["./bird.wav", "./cat.wav", "./tractor.wav"]

max_audio_len = 0
audio_list = []
for file in uploaded_files:
    wav, sr = librosa.load(file, sr=None)
    audio_list.append((wav, sr))

    max_audio_len = max(max_audio_len, wav.shape[0])

for i, audio in enumerate(audio_list, 1):
    st.write(f"Audio {i}")
    volume = st.slider("Volume (%)", 0, 100, 100, step=1, key=i)
    volume = volume / 100.0

    wav, sr = audio
    wav = wav * volume

    st.audio(create_audio_player(wav, sr))
    plot_wav(wav, max_audio_len)

    audio_segment = AudioSegment.from_file(create_audio_player(wav, sr))
    if i == 1:
        mixed_audio = audio_segment
    else:
        mixed_audio = mixed_audio.overlay(audio_segment, position=0) if mixed_audio.frame_count() > audio_segment.frame_count() else audio_segment.overlay(mixed_audio, position=0)

st.write("Mixed Audio")
st.audio(create_audio_player(np.array(mixed_audio.get_array_of_samples()), mixed_audio.frame_rate))

