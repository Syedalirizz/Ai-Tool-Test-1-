import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Initialize models
@st.cache_resource
def load_models():
    # Load Stable Diffusion using diffusers
    image_gen = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    image_gen.to("cuda") if torch.cuda.is_available() else image_gen.to("cpu")
    
    # Load a lightweight text generation model
    text_gen = pipeline("text-generation", model="gpt2")
    return image_gen, text_gen

# Load models
image_gen, text_gen = load_models()

# Title
st.title("AI Tool Prototype")

# Text Generation Section
st.header("Text Generation")
text_input = st.text_area("Enter text prompt for generation:")
if st.button("Generate Text"):
    if text_input:
        generated_text = text_gen(text_input, max_length=50, num_return_sequences=1)[0]['generated_text']
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.error("Please enter a text prompt.")

# Image Generation Section
st.header("Image Generation")
image_input = st.text_input("Enter a prompt for image generation:")
if st.button("Generate Image"):
    if image_input:
        with st.spinner("Generating image..."):
            generated_image = image_gen(image_input).images[0]
            st.subheader("Generated Image:")
            st.image(generated_image, use_column_width=True)
    else:
        st.error("Please enter a prompt for image generation.")
