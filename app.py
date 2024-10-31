import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Load models for various tasks
text_generator = pipeline("text-generation", model="gpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")
translator = pipeline("translation_en_to_fr")

# Stable Diffusion API details (replace with your Hugging Face token)
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
headers = {"Authorization": "Bearer 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"}

def query_stable_diffusion(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.content

# Streamlit app layout
st.title("AI Tools Hub")
st.subheader("Generate Captions, Summarize, Translate, Analyze Sentiment, and Generate Images")

# Caption Generator Section
st.header("Caption Generator")
user_input_caption = st.text_input("Enter a topic for caption generation:")
if st.button("Generate Caption"):
    if user_input_caption:
        caption = text_generator(user_input_caption, max_length=30)[0]['generated_text']
        st.write("Generated Caption:", caption)
    else:
        st.write("Please enter a topic for caption generation.")

# Image Generation Section
st.header("Image Generator")
user_input_image = st.text_input("Enter a prompt for image generation:")
if st.button("Generate Image"):
    if user_input_image:
        image_bytes = query_stable_diffusion(user_input_image)
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption=user_input_image)
        image.save("generated_image.png")
        st.download_button("Download Image", "generated_image.png")
    else:
        st.write("Please enter a prompt for image generation.")

# Text Summarization Section
st.header("Text Summarization")
user_input_summary = st.text_area("Enter text for summarization:")
if st.button("Summarize"):
    if user_input_summary:
        summary = summarizer(user_input_summary, max_length=50, min_length=25, do_sample=False)
        st.write("Summary:", summary[0]['summary_text'])
    else:
        st.write("Please enter text for summarization.")

# Translation Section
st.header("Text Translation")
user_input_translation = st.text_area("Enter English text to translate to French:")
if st.button("Translate"):
    if user_input_translation:
        translation = translator(user_input_translation)
        st.write("Translated Text:", translation[0]['translation_text'])
    else:
        st.write("Please enter text to translate.")

# Sentiment Analysis Section
st.header("Sentiment Analysis")
user_input_sentiment = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    if user_input_sentiment:
        sentiment = sentiment_analyzer(user_input_sentiment)
        st.write("Sentiment:", sentiment[0]['label'], "with score:", sentiment[0]['score'])
    else:
        st.write("Please enter text for sentiment analysis.")
