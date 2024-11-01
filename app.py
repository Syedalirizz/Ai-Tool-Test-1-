# Import necessary libraries
import streamlit as st
from transformers import pipeline, AutoModel

# Load Stability AI image generation model
image_generation_model = pipeline("image-generation", model="stabilityai/stable-diffusion-2-1-base")

# Load Liquid1 model
model = AutoModel.from_pretrained("Liquid1/llama-3-8b-liquid-coding-agent")

# Streamlit app layout
st.title("AI Tool")

# Input for image generation
st.header("Image Generation")
image_prompt = st.text_input("Enter a prompt for image generation:")
if st.button("Generate Image"):
    generated_image = image_generation_model(image_prompt)
    st.image(generated_image[0], caption="Generated Image")

# Input for text generation (if needed, adapt as necessary)
st.header("Text Generation")
user_input = st.text_input("Ask the AI:")
if st.button("Generate Response"):
    response = model(user_input)  # Adjust if using specific text generation pipeline
    st.write(response)

