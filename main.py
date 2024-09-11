import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_ace import st_ace
import json
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pydeck as pdk
import librosa
import librosa.display
import cv2
from PIL import Image
import io
import base64
import pedalboard
from pedalboard import Pedalboard, Chorus, Reverb
import mido
import pygame
import soundfile as sf
import sox
import os
import replicate

# Custom CSS (unchanged)
st.markdown("""
<style>
    /* ... (previous CSS remains unchanged) ... */
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation (unchanged)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to execute user-provided code (updated with new libraries and features)
def execute_code(code):
    # Create a dictionary of local variables that includes all imported libraries
    local_vars = {
        'st': st,
        'px': px,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'alt': alt,
        'pdk': pdk,
        'librosa': librosa,
        'cv2': cv2,
        'Image': Image,
        'io': io,
        'base64': base64,
        'pedalboard': pedalboard,
        'Pedalboard': Pedalboard,
        'Chorus': Chorus,
        'Reverb': Reverb,
        'mido': mido,
        'pygame': pygame,
        'sf': sf,
        'sox': sox,
        'replicate': replicate,
        'save_file': save_file,
        'load_file': load_file,
        'list_files': list_files
    }
    
    # Execute the code
    exec(code, globals(), local_vars)
    
    # Check if there's a matplotlib figure to display
    if 'plt' in local_vars and plt.get_fignums():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()
    
    # Check if there's a pygame surface to display
    if 'pygame' in local_vars and pygame.get_init():
        surface = pygame.display.get_surface()
        if surface:
            pygame_surface_to_image(surface)
    
    return local_vars

# Function to convert Pygame surface to Streamlit image (unchanged)
def pygame_surface_to_image(surface):
    buffer = surface.get_view("RGB")
    img = Image.frombytes("RGB", surface.get_size(), buffer.raw)
    st.image(img, caption="Pygame Output", use_column_width=True)

# New functions for file system management
def save_file(filename, content):
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    return os.listdir("generated_files")

# Function to call GPT-4 via requests (unchanged)
def chat_with_gpt(prompt, api_key, conversation_history):
    # ... (unchanged)

# Function to display chat messages (unchanged)
def display_chat_message(role, content):
    # ... (unchanged)

# Function to fix code (unchanged)
def fix_code(code, error_message, api_key):
    # ... (unchanged)

# Main function to run the Streamlit app (updated with new features)
def main():
    st.title("🚀 Advanced Coding Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your advanced coding assistant. How can I help you today? Feel free to ask questions, request code samples, or ask for explanations on various tasks including data visualization, audio processing, and image generation."})
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'last_code' not in st.session_state:
        st.session_state.last_code = None

    # Create directory for generated files if it doesn't exist
    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

    # Sidebar for API key inputs and library information
    with st.sidebar:
        st.header("Settings")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        replicate_api_key = st.text_input("Enter your Replicate API Key", type="password")
        st.markdown("---")
        st.markdown("### Quick Tips:")
        st.markdown("1. Chat naturally about coding tasks")
        st.markdown("2. Request code samples for various tasks")
        st.markdown("3. Experiment with data, audio, image, and AI models")
        st.markdown("4. Use buttons below chat to manage code and conversation")
        st.markdown("5. Access generated files using provided functions")
        
        st.markdown("---")
        st.markdown("### Available Libraries and Features:")
        st.markdown("- Plotting: matplotlib, seaborn, plotly, altair")
        st.markdown("- Data: pandas, numpy")
        st.markdown("- Geospatial: pydeck")
        st.markdown("- Audio: librosa, pedalboard, mido, soundfile, sox")
        st.markdown("- Image: PIL, cv2")
        st.markdown("- Game Dev: pygame")
        st.markdown("- AI Models: replicate")
        st.markdown("- File System: save_file, load_file, list_files")
        st.markdown("- Others: io, base64")

    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Code execution area
    st.markdown("### Code Execution Area")
    with st.container():
        st.markdown('<div class="code-execution-area">', unsafe_allow_html=True)
        
        # Display the current code
        if st.session_state.last_code:
            st.code(st.session_state.last_code, language="python")
        else:
            st.info("No code to display. Request a code sample or write some code to get started!")
            st.markdown("Here's an example to try:")
            example_code = """
# Example: Generate an image using Stable Diffusion via Replicate API
import replicate

# Set your Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "your_replicate_api_key_here"

# Run the model
output = replicate.run(
    "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    input={"prompt": "A futuristic city with flying cars"}
)

# Display the generated image
st.image(output[0], caption="Generated Image", use_column_width=True)

# Save the image URL
save_file("generated_image_url.txt", output[0])
st.write("Image URL saved to 'generated_image_url.txt'")
"""
            st.code(example_code, language="python")
            if st.button("Try This Example"):
                st.session_state.last_code = example_code

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask me anything about coding or request a visualization...")

    # Action buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🏃‍♂️ Run Code", key="run_code"):
            if st.session_state.last_code:
                with st.spinner("Executing code..."):
                    try:
                        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
                        result = execute_code(st.session_state.last_code)
                        st.success("Code executed successfully.")
                        st.session_state.last_error = None
                    except Exception as e:
                        error_msg = f"Error executing code: {str(e)}"
                        st.error(error_msg)
                        st.session_state.last_error = str(e)
            else:
                st.warning("No code to execute. Please request a code sample first.")

    with col2:
        if st.button("🔧 Fix and Rerun", key="fix_and_rerun"):
            if st.session_state.last_error and st.session_state.last_code:
                with st.spinner("Fixing code..."):
                    fixed_code = fix_code(st.session_state.last_code, st.session_state.last_error, openai_api_key)
                    st.session_state.last_code = fixed_code
            else:
                st.warning("No error to fix or no previous code execution. Please run some code first.")

    with col3:
        if st.button("🧹 Clear Code", key="clear_code"):
            st.session_state.last_code = None
            st.session_state.last_error = None

    with col4:
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Chat cleared. How can I assist you?"})

    with col5:
        if st.button("🔄 Reset All", key="reset_all"):
            st.session_state.messages = []
            st.session_state.last_error = None
            st.session_state.last_code = None
            st.session_state.messages.append({"role": "assistant", "content": "Everything has been reset. How can I help you today?"})

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        # Process with GPT-4
        if openai_api_key:
            with st.spinner("Thinking..."):
                response = chat_with_gpt(prompt, openai_api_key, st.session_state.messages[:-1])
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                # Update last_code if the response contains a code block
                if "```python" in response:
                    st.session_state.last_code = response.split("```python")[1].split("```")[0].strip()
        else:
            st.warning("Please enter a valid OpenAI API key in the sidebar.")

    # Display generated files
    st.markdown("### Generated Files")
    files = list_files()
    if files:
        for file in files:
            if st.button(f"View {file}"):
                content = load_file(file)
                st.text_area("File Content", content, height=200)
    else:
        st.info("No generated files yet.")

# Entry point
if __name__ == "__main__":
    main()
