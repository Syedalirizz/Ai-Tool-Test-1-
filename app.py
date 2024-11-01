import streamlit as st
from transformers import pipeline

# Initialize the models
@st.cache_resource
def load_models():
    # Load image generation model (Stable Diffusion)
    image_gen = pipeline("image-generation", model="CompVis/stable-diffusion-v1-4")
    # Load a simple text generation model
    text_gen = pipeline("text-generation", model="gpt2")  # Using GPT-2 for text generation
    return image_gen, text_gen

# Load models
image_gen, text_gen = load_models()

# Title
st.title("AI Tool Prototype")

# Input section for text generation
st.header("Text Generation")
text_input = st.text_area("Enter text prompt for generation:")
if st.button("Generate Text"):
    if text_input:
        generated_text = text_gen(text_input, max_length=50, num_return_sequences=1)[0]['generated_text']
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.error("Please enter a text prompt.")

# Input section for image generation
st.header("Image Generation")
image_input = st.text_input("Enter a prompt for image generation:")
if st.button("Generate Image"):
    if image_input:
        generated_image = image_gen(image_input)[0]['images'][0]
        st.subheader("Generated Image:")
        st.image(generated_image, use_column_width=True)
    else:
        st.error("Please enter a prompt for image generation.")

