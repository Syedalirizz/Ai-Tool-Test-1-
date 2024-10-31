import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Define Streamlit app
st.set_page_config(page_title="AI Tool", page_icon=":sparkles:", layout="wide")

# Load transformers models with explicit names
text_generator = pipeline("text-generation", model="gpt2")
image_generator = pipeline("image-generation", model="CompVis/stable-diffusion-v-1-4")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f5;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .header {
        background-color: #6200ea;
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .button {
        background-color: #6200ea;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #3700b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create header
st.markdown("<div class='header'><h1>AI Tool</h1></div>", unsafe_allow_html=True)

# User input for text generation
st.subheader("Generate Text")
user_input = st.text_area("Enter your prompt:")
if st.button("Generate"):
    output = text_generator(user_input, max_length=100)[0]['generated_text']
    st.write("Generated Text:")
    st.write(output)

# User input for image generation
st.subheader("Generate Image")
image_prompt = st.text_input("Enter image description:")
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = image_generator(image_prompt)
        st.image(image, caption=image_prompt, use_column_width=True)

# User input for summarization
st.subheader("Summarize Text")
text_to_summarize = st.text_area("Enter text to summarize:")
if st.button("Summarize"):
    summary = summarizer(text_to_summarize, max_length=50, min_length=25, do_sample=False)
    st.write("Summary:")
    st.write(summary[0]['summary_text'])

# User input for sentiment analysis
st.subheader("Analyze Sentiment")
sentiment_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze"):
    sentiment = sentiment_analyzer(sentiment_input)
    st.write("Sentiment:")
    st.write(sentiment)

# User input for translation
st.subheader("Translate Text (English to French)")
text_to_translate = st.text_area("Enter text to translate:")
if st.button("Translate"):
    translation = translator(text_to_translate)
    st.write("Translation:")
    st.write(translation[0]['translation_text'])

# Footer
st.markdown("<div class='footer'><p>AI Tool - All rights reserved</p></div>", unsafe_allow_html=True)
