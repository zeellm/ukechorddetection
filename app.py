import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import keras 
from keras.models import load_model
from skimage.transform import resize
from chords import predict_file
import tensorflow
import tempfile
from audiorecorder import audiorecorder
import pydub 
from st_audiorec import st_audiorec
import soundfile


# Load the pre-trained Keras model
# model = load_model("keras_model.h5")


def record_audio(duration):
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return recording.flatten(), fs

# def classify_chord(audio_data, model):
#     # Convert audio data to Mel spectrogram
#     mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=44100)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    
#     # Resize Mel spectrogram to match model input shape
#     mel_spec_db_resized = resize(mel_spec_db, (128, 5000), anti_aliasing=True)
    
#     # Reshape to match model input shape (None, 5000)
#     mel_spec_db_resized = np.expand_dims(mel_spec_db_resized, axis=0)  # Add batch dimension
    
#     # Predict chord
#     prediction = model.predict(mel_spec_db_resized)
#     predicted_label = np.argmax(prediction)
#     confidence = prediction[0][predicted_label]

#     return predicted_label, confidence


st.set_page_config(page_title='UkeLEARN', page_icon='🎸')

st.title("String Instrument Learner")

option = st.radio("Choose how to input the chord:", ("Record Chord", "Upload Chord File"))

if option == "Record Chord":
    # if st.button("Record"):
        # audio = audiorecorder("Click to record","Click to stop recording")
        # if len(audio) > 0: 
        #     st.audio(audio.export().read())  
        #     audio.export("audio.wav", format="wav")

        #     # To get audio properties, use pydub AudioSegment properties:
        #     st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

        #     # Export the recorded audio to a temporary WAV file
        #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        #             audio.export(temp_file.name, format="wav")
        #             temp_file_path = temp_file.name
                
            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                st.audio(wav_audio_data, format='audio/wav')

                with st.spinner('Processing...'):
                    file_path = 'output_audio.wav'
                    with open(file_path, 'wb') as f:
                        f.write(wav_audio_data)

                    st.write("Recording finished!")
                
                chord_names = ["A", "C", "E"]
                st.write("Classifying...")
                data, samplerate = soundfile.read(file_path)
                soundfile.write(file_path, data, samplerate, subtype='PCM_24')
                temp_file_path = file_path
                
                chord_label = predict_file([temp_file_path])
                print(chord_label)
                st.write(f"Detected Chord: {chord_label}")
        
elif option == "Upload Chord File":
    uploaded_file = st.file_uploader("Upload your chord file (WAV format)", type="wav")

    if uploaded_file is not None:
        with st.spinner('Processing...'):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            st.audio(temp_file_path, format='audio/wav')
            st.write("File uploaded successfully!")
        
        st.write("Classifying...")
        chord_label = predict_file([temp_file_path])
        print(chord_label)
        st.write("Detected Chord: " + chord_label)