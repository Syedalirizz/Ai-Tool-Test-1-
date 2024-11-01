import streamlit as st
import requests
import io
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch

# Load Longformer model and tokenizer
longformer_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
longformer_model = AutoModel.from_pretrained("allenai/longformer-base-4096")

# API URL and headers for Stable Diffusion
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
headers = {"Authorization": "Bearer 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"}

# Function to query Stable Diffusion
def query_stable_diffusion(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.content

# Streamlit UI
st.title('AI Tools Suite: Image Generation and Long Document Processing')

# Image Generator Section
st.header("Image Generator")
user_input_image = st.text_input("Enter a prompt for image generation:")
if st.button('Generate Image'):
    image_bytes = query_stable_diffusion(user_input_image)
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption=user_input_image)
    image.save("generated_image.png")
    st.download_button("Download Image", "generated_image.png")

# Long Document Processing Section
st.header("Long Document Processing")
user_input_long_doc = st.text_area("Enter a long document for processing:")
if st.button('Process Document'):
    inputs = longformer_tokenizer(user_input_long_doc, return_tensors="pt", max_length=4096, truncation=True)
    with torch.no_grad():
        outputs = longformer_model(**inputs)
    st.write("Processed Outputs:", outputs)

