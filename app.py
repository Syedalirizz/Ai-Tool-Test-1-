import streamlit as st
from transformers import AutoModel, AutoTokenizer, pipeline
import requests
from PIL import Image
import io

# Load Llama-3 coding agent model
tokenizer = AutoTokenizer.from_pretrained("Liquid1/llama-3-8b-liquid-coding-agent")
model = AutoModel.from_pretrained("Liquid1/llama-3-8b-liquid-coding-agent")

# Initialize Stability AI image generation pipeline
stability_pipeline = pipeline("image-generation", model="stabilityai/stable-diffusion-2-1-base")

# Streamlit app title
st.title("AI Tool Platform")

# Input text for Llama-3
user_input = st.text_area("Enter your prompt for coding assistance:")

if st.button("Get Coding Help"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Llama-3 Response:")
        st.write(response)
    else:
        st.error("Please enter a prompt.")

# Input text for image generation
image_prompt = st.text_input("Enter a prompt for image generation:")

if st.button("Generate Image"):
    if image_prompt:
        image = stability_pipeline(image_prompt)[0]
        image_url = image['url']
        st.image(image_url, caption="Generated Image")
    else:
        st.error("Please enter an image prompt.")
