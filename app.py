import streamlit as st
import requests
import io
from PIL import Image

# API URLs and headers
stability_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
whisper_api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
headers = {"Authorization": "Bearer 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"}

# Stability AI Image Generation
def generate_image(prompt):
    response = requests.post(stability_api_url, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        st.error("Error generating image.")
        return None

# OpenAI Whisper Transcription
def transcribe_audio(file):
    with open(file, "rb") as f:
        audio_data = f.read()
    response = requests.post(whisper_api_url, headers=headers, data=audio_data)
    if response.status_code == 200:
        return response.json().get("text", "Transcription not available.")
    else:
        st.error("Error transcribing audio.")
        return None

# Streamlit UI
st.title("AI Tools Suite: Image Generation and Audio Transcription")

# Image Generation Section
st.header("Image Generation (Stability AI)")
prompt = st.text_input("Enter a prompt for image generation:")
if st.button("Generate Image"):
    if prompt:
        generated_image = generate_image(prompt)
        if generated_image:
            st.image(generated_image, caption=prompt)
            st.download_button("Download Image", data=generated_image.tobytes(), file_name="generated_image.png")
    else:
        st.warning("Please enter a prompt.")

# Audio Transcription Section
st.header("Audio Transcription (OpenAI Whisper)")
uploaded_file = st.file_uploader("Upload an audio file (e.g., .flac format) for transcription:")
if st.button("Transcribe Audio"):
    if uploaded_file is not None:
        transcription = transcribe_audio(uploaded_file)
        st.write("Transcription:", transcription)
    else:
        st.warning("Please upload an audio file.")
