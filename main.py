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
import contextlib
import plotly.graph_objs as go
import uuid

# Custom CSS (unchanged)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    }
    .chat-message.user {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .chat-message.assistant {
        background-color: rgba(0, 0, 0, 0.1);
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    .floating-button {
        position: fixed;
        right: 20px;
        bottom: 20px;
    }
    .code-block {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .code-block pre {
        margin-bottom: 0;
    }
    .error-message {
        background-color: rgba(255, 0, 0, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .fix-button {
        background-color: #FFA500;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 5px 10px;
        transition: all 0.3s ease;
    }
    .fix-button:hover {
        background-color: #FF8C00;
        transform: scale(1.05);
    }
    .code-execution-area {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation (unchanged)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Updated execute_code function
def execute_code(code):
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
        'save_file': save_file,
        'load_file': load_file,
        'list_files': list_files
    }
    
    output = io.StringIO()
    
    try:
        compile(code, '<string>', 'exec')
        
        with contextlib.redirect_stdout(output):
            exec(code, globals(), local_vars)
        
        if output.getvalue():
            st.text("Print output:")
            st.code(output.getvalue(), language="")
        
        if 'plt' in local_vars and plt.get_fignums():
            fig = plt.gcf()
            st.pyplot(fig)
            save_plot(fig, "matplotlib_plot.png")
            plt.close()
        
        if 'fig' in local_vars and isinstance(local_vars['fig'], go.Figure):
            st.plotly_chart(local_vars['fig'])
            save_plotly(local_vars['fig'], "plotly_plot.html")
        
        if 'pygame' in local_vars and pygame.get_init():
            surface = pygame.display.get_surface()
            if surface:
                img = pygame_surface_to_image(surface)
                st.image(img, caption="Pygame Output", use_column_width=True)
                save_image(img, "pygame_output.png")
        
        if 'audio' in local_vars and isinstance(local_vars['audio'], np.ndarray):
            st.audio(local_vars['audio'], sample_rate=local_vars.get('sample_rate', 44100))
            save_audio(local_vars['audio'], local_vars.get('sample_rate', 44100), "audio_output.wav")
        
        # Check for PIL Image
        if 'image' in local_vars and isinstance(local_vars['image'], Image.Image):
            st.image(local_vars['image'], caption="Generated Image", use_column_width=True)
            save_image(local_vars['image'], "generated_image.png")
        
        return True, "Code executed successfully."
    except SyntaxError as e:
        error_msg = f"Syntax Error: {str(e)}"
        st.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        return False, error_msg

# Function to convert Pygame surface to PIL Image
def pygame_surface_to_image(surface):
    buffer = surface.get_view("RGB")
    return Image.frombytes("RGB", surface.get_size(), buffer.raw)

# Functions for file system management
def save_file(filename, content):
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    return os.listdir("generated_files")

# New functions to save various outputs
def save_plot(fig, filename):
    fig.savefig(f"generated_files/{filename}")

def save_plotly(fig, filename):
    fig.write_html(f"generated_files/{filename}")

def save_image(img, filename):
    img.save(f"generated_files/{filename}")

def save_audio(audio, sample_rate, filename):
    sf.write(f"generated_files/{filename}", audio, sample_rate)

# Function to call GPT-4 via requests (unchanged)
def chat_with_gpt(prompt, api_key, conversation_history):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = conversation_history + [{"role": "user", "content": prompt}]
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 2000
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display chat messages (unchanged)
def display_chat_message(role, content):
    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
        else:
            if "```python" in content:
                parts = content.split("```python")
                st.markdown(parts[0])
                st.code(parts[1].split("```")[0], language="python")
                if len(parts) > 2:
                    st.markdown(parts[2])
            else:
                st.markdown(content)

# Function to fix code (unchanged)
def fix_code(code, error_message, api_key):
    prompt = f"The following Python code produced an error:\n\n```python\n{code}\n```\n\nError message: {error_message}\n\nPlease provide a corrected version of the code that fixes this error."
    fixed_code = chat_with_gpt(prompt, api_key, [])
    return fixed_code

# Main function to run the Streamlit app
def main():
    st.title("🚀 Advanced Coding Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your advanced coding assistant. How can I help you today? Feel free to ask questions, request code samples, or ask for explanations on various tasks including data visualization and audio processing."})
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'last_code' not in st.session_state:
        st.session_state.last_code = None

    # Create directory for generated files if it doesn't exist
    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

    # Sidebar for API key input and library information
    with st.sidebar:
        st.header("Settings")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        st.markdown("---")
        st.markdown("### Quick Tips:")
        st.markdown("1. Chat naturally about coding tasks")
        st.markdown("2. Request code samples for various tasks")
        st.markdown("3. Experiment with data visualization and audio processing")
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
# Example: Create a smiley face image
from PIL import Image, ImageDraw

# Create a new image with a white background
image = Image.new('RGB', (200, 200), color='white')

# Create a drawing object
draw = ImageDraw.Draw(image)

# Draw the face
draw.ellipse([20, 20, 180, 180], outline='black', width=2)

# Draw the eyes
draw.ellipse([55, 65, 85, 95], fill='black')
draw.ellipse([115, 65, 145, 95], fill='black')

# Draw the smile
draw.arc([50, 85, 150, 155], start=0, end=180, fill='black', width=2)

# Display the image
st.image(image, caption='Smiley Face', use_column_width=True)

# Save the image
image.save('generated_files/smiley_face.png')
st.write("Image saved as 'smiley_face.png'")
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
                    success, message = execute_code(st.session_state.last_code)
                    if success:
                        st.success(message)
                        st.session_state.last_error = None
                    else:
                        st.session_state.last_error = message
            else:
                st.warning("No code to execute. Please request a code sample first.")

    with col2:
        if st.button("🔧 Fix and Rerun", key="fix_and_rerun"):
            if st.session_state.last_error and st.session_state.last_code:
                with st.spinner("Fixing code..."):
                    fixed_code = fix_code(st.session_state.last_code, st.session_state.last_error, openai_api_key)
                    st.session_state.last_code = fixed_code
                    success, message = execute_code(fixed_code)
                    if success:
