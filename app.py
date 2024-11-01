import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
from PIL import Image
import io

# Load alternative coding model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

# Initialize Stability AI image generation pipeline
stability_pipeline = pipeline("image-generation", model="stabilityai/stable-diffusion-2-1-base")

# Streamlit app title
st.title("AI Tool Platform")

# Input text for coding assistance
user_input = st.text_area("Enter your prompt for coding assistance:")

if st.button("Get Coding Help"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Response:")
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
