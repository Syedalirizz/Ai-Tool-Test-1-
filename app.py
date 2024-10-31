import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Set your Hugging Face token here
HUGGINGFACE_TOKEN = "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"

# Load pipelines with token
text_gen_pipeline = pipeline("text-generation", model="gpt2", use_auth_token=HUGGINGFACE_TOKEN)
image_gen_pipeline = pipeline("image-generation", model="CompVis/stable-diffusion-v-1-4", use_auth_token=HUGGINGFACE_TOKEN)
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=HUGGINGFACE_TOKEN)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=HUGGINGFACE_TOKEN)
translation_pipeline = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", use_auth_token=HUGGINGFACE_TOKEN)

# Set up the Streamlit app layout
st.set_page_config(page_title="AI Tools", layout="wide")
st.title("AI Tools")
st.markdown("<style>body{background-color: #f0f0f5;}</style>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Select an option:", ["Home", "Generate Image", "Text Generation", "Summarization", "Sentiment Analysis", "Translation"])

# Home section
if option == "Home":
    st.subheader("Welcome to the AI Tools App")
    st.write("This app allows you to generate images, create text, summarize content, analyze sentiment, and translate text.")

# Image Generation section
elif option == "Generate Image":
    st.subheader("Generate Image")
    image_prompt = st.text_input("Enter a prompt for image generation:")
    if st.button("Generate Image"):
        if image_prompt:
            with st.spinner("Generating image..."):
                image = image_gen_pipeline(image_prompt)[0]['image']
                st.image(image, caption="Generated Image", use_column_width=True)

# Text Generation section
elif option == "Text Generation":
    st.subheader("Text Generation")
    text_prompt = st.text_area("Enter a prompt for text generation:")
    if st.button("Generate Text"):
        if text_prompt:
            with st.spinner("Generating text..."):
                generated_text = text_gen_pipeline(text_prompt, max_length=100)[0]['generated_text']
                st.write(generated_text)

# Summarization section
elif option == "Summarization":
    st.subheader("Summarization")
    summary_input = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if summary_input:
            with st.spinner("Summarizing text..."):
                summary = summarization_pipeline(summary_input, max_length=50)[0]['summary_text']
                st.write(summary)

# Sentiment Analysis section
elif option == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    sentiment_input = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if sentiment_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = sentiment_pipeline(sentiment_input)[0]
                st.write(f"Label: {sentiment_result['label']}, Score: {sentiment_result['score']:.2f}")

# Translation section
elif option == "Translation":
    st.subheader("Translation")
    translation_input = st.text_area("Enter text in English to translate to French:")
    if st.button("Translate"):
        if translation_input:
            with st.spinner("Translating..."):
                translated_text = translation_pipeline(translation_input)[0]['translation_text']
                st.write(translated_text)

# Add some space and styling
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<footer>Created by Your Name</footer>", unsafe_allow_html=True)
