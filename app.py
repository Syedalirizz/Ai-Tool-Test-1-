import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline

# Initialize the models
@st.cache_resource
def load_models():
    # Load the Llama model
    tokenizer = AutoTokenizer.from_pretrained("Liquid1/llama-3-8b-liquid-coding-agent")
    model = AutoModelForCausalLM.from_pretrained("Liquid1/llama-3-8b-liquid-coding-agent")
    
    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    pipe.to("cuda")  # Use GPU if available

    return tokenizer, model, pipe

tokenizer, model, sd_pipe = load_models()

# Streamlit interface
st.title("AI Tools Application")

# Text Generation
st.header("Text Generation")
input_text = st.text_area("Enter your prompt for text generation:")
if st.button("Generate Text"):
    if input_text:
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(generated_text)
    else:
        st.error("Please enter a prompt.")

# Image Generation
st.header("Image Generation")
image_prompt = st.text_input("Enter your prompt for image generation:")
if st.button("Generate Image"):
    if image_prompt:
        with st.spinner("Generating image..."):
            generated_image = sd_pipe(image_prompt)["sample"][0]
            st.image(generated_image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a prompt.")

# Running the Streamlit application
if __name__ == "__main__":
    st.run()
