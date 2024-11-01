import streamlit as st
import requests
import io
from PIL import Image
from transformers import WhisperProcessor
import torch

# API URLs and headers
image_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
audio_api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
headers = {"Authorization": "Bearer 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"}

# Initialize WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

# Image Generation Function
def generate_image(prompt):
    response = requests.post(image_api_url, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        st.error("Error generating image.")
        return None

# Audio Transcription Function
def transcribe_audio(uploaded_file):
    if uploaded_file is not None:
        audio_data = uploaded_file.read()  # Read the file as bytes
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        
        # Call the Whisper API with processed inputs
        response = requests.post(audio_api_url, headers=headers, data=inputs["input_features"].numpy().tobytes())
        if response.status_code == 200:
            return response.json().get("text", "Transcription not available.")
        else:
            st.error("Error transcribing audio. Status code: " + str(response.status_code))
            return None
    else:
        st.error("No file uploaded.")
        return None

# Streamlit UI
st.title("AI Tools Suite")

# Image Generation Section
st.header("Generate an Image")
prompt = st.text_input("Enter a prompt:")
if st.button("Generate Image"):
    if prompt:
        generated_image = generate_image(prompt)
        if generated_image:
            st.image(generated_image, caption=prompt)
            st.download_button("Download Image", data=generated_image.tobytes(), file_name="generated_image.png")
    else:
        st.warning("Please enter a prompt.")

# Audio Transcription Section
st.header("Transcribe Audio")
uploaded_file = st.file_uploader("Upload an audio file (e.g., .flac format):")
if uploaded_file is not None and st.button("Transcribe Audio"):
    transcription = transcribe_audio(uploaded_file)
    if transcription:
        st.write("Transcription:", transcription)
    else:
        st.warning("Unable to generate transcription.")
