import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
import io

# Set up Hugging Face token
HUGGINGFACE_TOKEN = "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"  # Your token

# Load pipelines
text_gen_pipeline = pipeline("text-generation", model="gpt2", use_auth_token=HUGGINGFACE_TOKEN)
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=HUGGINGFACE_TOKEN)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=HUGGINGFACE_TOKEN)
translation_pipeline = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", use_auth_token=HUGGINGFACE_TOKEN)

# Image generation function
def generate_image(prompt):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.content
    else:
        st.error("Error generating image!")
        return None

# Streamlit app layout
st.title("AI Tool Suite")
st.sidebar.title("Options")

# Text Generation
if st.sidebar.button("Generate Text"):
    prompt = st.sidebar.text_input("Enter text prompt:")
    if prompt:
        generated_text = text_gen_pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.write("Generated Text:")
        st.write(generated_text)

# Image Generation
if st.sidebar.button("Generate Image"):
    image_prompt = st.sidebar.text_input("Enter image prompt:")
    if image_prompt:
        image_bytes = generate_image(image_prompt)
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=image_prompt)

# Summarization
if st.sidebar.button("Summarize Text"):
    text_to_summarize = st.sidebar.text_area("Enter text to summarize:")
    if text_to_summarize:
        summary = summarization_pipeline(text_to_summarize, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        st.write("Summary:")
        st.write(summary)

# Sentiment Analysis
if st.sidebar.button("Analyze Sentiment"):
    sentiment_text = st.sidebar.text_area("Enter text for sentiment analysis:")
    if sentiment_text:
        sentiment = sentiment_pipeline(sentiment_text)[0]
        st.write("Sentiment:", sentiment['label'], "with a score of", sentiment['score'])

# Translation
if st.sidebar.button("Translate Text"):
    text_to_translate = st.sidebar.text_area("Enter text to translate:")
    if text_to_translate:
        translated_text = translation_pipeline(text_to_translate)[0]['translation_text']
        st.write("Translated Text:")
        st.write(translated_text)
