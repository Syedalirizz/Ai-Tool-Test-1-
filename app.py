import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import requests

# Hugging Face Token
HUGGINGFACE_TOKEN = "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"

# Load pipelines only when called
def load_model(model_name):
    return pipeline(model_name, use_auth_token=HUGGINGFACE_TOKEN)

# Function to query image generation
def generate_image(prompt):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.content if response.status_code == 200 else None

# Streamlit app layout
st.title("AI Tool Suite")
st.sidebar.title("Options")

# Text Generation
if st.sidebar.button("Generate Text"):
    text_gen_pipeline = load_model("gpt2")  # Load on demand
    prompt = st.sidebar.text_input("Enter text prompt:")
    if prompt:
        try:
            generated_text = text_gen_pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            st.write("Generated Text:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"Error generating text: {e}")

# Image Generation
if st.sidebar.button("Generate Image"):
    image_prompt = st.sidebar.text_input("Enter image prompt:")
    if image_prompt:
        try:
            image_bytes = generate_image(image_prompt)
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption=image_prompt)
            else:
                st.error("Failed to generate image.")
        except Exception as e:
            st.error(f"Error generating image: {e}")

# Text Summarization
if st.sidebar.button("Summarize Text"):
    summarization_pipeline = load_model("facebook/bart-large-cnn")  # Using BART for summarization
    text_to_summarize = st.sidebar.text_area("Enter text to summarize:")
    if text_to_summarize:
        try:
            summary = summarization_pipeline(text_to_summarize, max_length=50, min_length=25, do_sample=False)
            st.write("Summary:")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error summarizing text: {e}")

# Sentiment Analysis
if st.sidebar.button("Analyze Sentiment"):
    sentiment_pipeline = load_model("distilbert-base-uncased-finetuned-sst-2-english")  # Using DistilBERT for sentiment
    text_to_analyze = st.sidebar.text_area("Enter text for sentiment analysis:")
    if text_to_analyze:
        try:
            sentiment = sentiment_pipeline(text_to_analyze)
            st.write("Sentiment Analysis Result:")
            st.write(sentiment)
        except Exception as e:
            st.error(f"Error analyzing sentiment: {e}")

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f5;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)
