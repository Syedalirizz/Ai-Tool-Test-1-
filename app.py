import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline

# Function to load models
@st.cache_resource
def load_models():
    # Load a lightweight text generation model (GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Load the Stable Diffusion model for image generation
    sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    sd_pipe.to("cuda")  # Use GPU if available

    return tokenizer, model, sd_pipe

# Load models
tokenizer, model, sd_pipe = load_models()

# Streamlit application interface
st.title("AI Tools Prototype")

# Text Generation Section
st.header("Text Generation")
input_text = st.text_area("Enter your prompt for text generation:")
if st.button("Generate Text"):
    if input_text:
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)  # Generate up to 50 tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Text:")
        st.write(generated_text)  # Display generated text
    else:
        st.error("Please enter a prompt.")

# Image Generation Section
st.header("Image Generation")
image_prompt = st.text_input("Enter your prompt for image generation:")
if st.button("Generate Image"):
    if image_prompt:
        with st.spinner("Generating image..."):
            generated_image = sd_pipe(image_prompt)["sample"][0]
            st.subheader("Generated Image:")
            st.image(generated_image, caption="Generated Image", use_column_width=True)  # Display generated image
    else:
        st.error("Please enter a prompt.")

# Footer for the app
st.markdown("""
    ### About This App
    This application utilizes state-of-the-art AI models for text and image generation.
    - **Text Generation:** Powered by GPT-2, generates coherent text based on your prompt.
    - **Image Generation:** Utilizes Stable Diffusion to create images based on descriptive prompts.
""")

# Running the Streamlit application
if __name__ == "__main__":
    st.run()
