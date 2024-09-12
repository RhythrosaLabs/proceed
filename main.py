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

# Custom CSS
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
        object-fit: cover.
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
        transition: all 0.3s ease.
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
        margin-bottom: 20px.
    }
</style>
""", unsafe_allow_html=True)

# Function to execute user-provided code
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
    
    exec(code, globals(), local_vars)

    # Check if there's a matplotlib figure to display
    if 'plt' in local_vars and plt.get_fignums():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()
    
    if 'pygame' in local_vars and pygame.get_init():
        surface = pygame.display.get_surface()
        if surface:
            pygame_surface_to_image(surface)
    
    return local_vars

# Function to convert Pygame surface to Streamlit image
def pygame_surface_to_image(surface):
    buffer = surface.get_view("RGB")
    img = Image.frombytes("RGB", surface.get_size(), buffer.raw)
    st.image(img, caption="Pygame Output", use_column_width=True)

# File management functions
def save_file(filename, content):
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    return os.listdir("generated_files")

# Main function
def main():
    st.title("🚀 Advanced Coding Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your advanced coding assistant. How can I help you today?"})
    if 'last_code' not in st.session_state:
        st.session_state.last_code = ""

    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

    # Sidebar for API key input
    with st.sidebar:
        st.header("Settings")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        st.markdown("---")
    
    # Layout for chat and code execution
    col1, col2 = st.columns([1, 1.5])

    with col1:
        # Chat input and history
        for message in st.session_state.messages:
            role, content = message["role"], message["content"]
            with st.chat_message(role):
                st.markdown(content)

        # User input for chat
        prompt = st.chat_input("Ask me anything or provide code to run...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Handle user response from GPT if applicable (not implemented here)

    with col2:
        # Editable code execution area
        st.markdown("### Code Execution Area")
        st.session_state.last_code = st_ace(
            value=st.session_state.last_code,
            language="python",
            theme="monokai",
            key="code_editor"
        )

        # Buttons for code execution and file management
        if st.button("🏃‍♂️ Run Code", key="run_code"):
            try:
                execute_code(st.session_state.last_code)
                st.success("Code executed successfully.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

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
