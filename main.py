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
import time
import psutil

# Set page config for wide layout
st.set_page_config(page_title="🚀 Code Assistant", page_icon="🚀", layout="wide")

# Custom CSS for sleek design and responsive layout
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
    .file-display {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.9);
    }
    .code-execution-area {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .code-execution-area pre {
        margin-bottom: 0;
    }
    .bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: rgba(49, 51, 63, 0.9);
        padding: 10px;
        z-index: 999;
        display: flex;
        justify-content: space-around;
    }
    .bottom-bar .stButton > button {
        height: 2.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .main .block-container {
        padding-bottom: 5rem;
    }
    .stChatInputContainer {
        position: fixed;
        bottom: 4rem;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: rgba(49, 51, 63, 0.9);
        z-index: 998;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper functions for loading Lottie animations, saving/loading files
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def ensure_directory_exists():
    os.makedirs("generated_files", exist_ok=True)

def save_file(filename, content):
    ensure_directory_exists()
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    ensure_directory_exists()
    return os.listdir("generated_files")

def display_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif']:
        st.image(file_path, use_column_width=True)
    elif ext in ['.wav', '.mp3']:
        st.audio(file_path)
    elif ext == '.mp4':
        st.video(file_path)
    elif ext == '.html':
        with open(file_path, 'r') as f:
            st.components.v1.html(f.read(), height=600)
    else:
        st.code(load_file(file_path), language="python")

# Function to execute Python code with visualization support
def execute_code(code):
    local_vars = {'st': st, 'px': px, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'alt': alt, 'pdk': pdk, 'librosa': librosa, 'cv2': cv2, 'Image': Image}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals(), local_vars)
        st.code(output.getvalue(), language="")
        # Handle common output types
        if 'fig' in local_vars and isinstance(local_vars['fig'], go.Figure):
            st.plotly_chart(local_vars['fig'])
        elif plt.get_fignums():
            st.pyplot(plt.gcf())
            plt.close()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Function to chat with GPT-4 via API
def chat_with_gpt(prompt, api_key, conversation_history):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "gpt-4", "messages": conversation_history + [{"role": "user", "content": prompt}], "max_tokens": 2000}
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display chat messages
def display_chat_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

# Main function to run the app
def main():
    st.title("🚀 Advanced Coding Assistant")
    
    # Initialize session state for messages and code
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your coding assistant. How can I assist you today?"}]
    if 'code_editor' not in st.session_state:
        st.session_state.code_editor = ""
    
    # Sidebar with settings
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask for code, visualizations, or any coding help...")
    if prompt:
        if api_key:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = chat_with_gpt(prompt, api_key, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if "```python" in response:
                code_block = response.split("```python")[1].split("```")[0].strip()
                st.session_state.code_editor = code_block
            st.experimental_rerun()
        else:
            st.warning("Please enter an OpenAI API key.")
    
    # Code editor section
    st.subheader("Code Editor & Executor")
    st.session_state.code_editor = st_ace(
        value=st.session_state.code_editor, language="python", theme="monokai", height=300,
        keybinding="vscode", font_size=14, tab_size=4, show_gutter=True
    )
    
    # Buttons for executing code, fixing, and clearing
    if st.button("🏃‍♂️ Run Code"):
        execute_code(st.session_state.code_editor)
    
    # Display generated files if any
    st.subheader("Generated Files")
    files = list_files()
    if files:
        for file in files:
            with st.expander(file):
                display_file(f"generated_files/{file}")
    else:
        st.info("No files generated yet.")
    
    # Bottom bar with action buttons
    st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("🔄 Reset All", on_click=lambda: st.session_state.clear())
    with col2:
        st.button("🧹 Clear Code", on_click=lambda: st.session_state.update({'code_editor': ""}))
    with col3:
        st.button("🏃‍♂️ Run Code", on_click=lambda: execute_code(st.session_state.code_editor))
    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
