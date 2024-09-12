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
import multiprocessing
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

# Custom CSS with added styles for bottom bar and chat input
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
    .file-display {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.9);
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
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
    .file-display {
        border: 1px solid #ccc;
        border-radius: 10px;
        margin: 5px 0;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .file-display h6 {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    /* New styles for the bottom action bar */
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
        align-items: center;
    }
    .bottom-bar .stButton > button {
        height: 2.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Adjust main content to not be hidden behind bottom bar */
    .main .block-container {
        padding-bottom: 5rem;
    }
    /* Custom styles for the chat input */
    .stChatInputContainer {
        position: fixed;
        bottom: 4rem;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: rgba(49, 51, 63, 0.9);
        z-index: 998;
    }
    .stChatInputContainer > div {
        margin-bottom: 0 !important;
    }
    /* Hide default Streamlit watermark */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
def clear_generated_files():
    folder = "generated_files"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)


# Updated function to execute user-provided code
# Function to execute user-provided code
# Function to execute user-provided code
def execute_code(code, timeout=30):
    def worker(code, return_dict):
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
            'save_file': save_file,
            'load_file': load_file,
            'list_files': list_files,
            'save_image': save_image,
            'save_audio': save_audio,
            'display_generated_file': display_generated_file,
        }

        output = io.StringIO()
        error_output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
                exec(code, globals(), local_vars)

            return_dict['output'] = output.getvalue()
            return_dict['error'] = error_output.getvalue()
        except Exception:
            return_dict['error'] = traceback.format_exc()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(target=worker, args=(code, return_dict))

    process.start()
    start_time = time.time()

    while process.is_alive():
        if time.time() - start_time > timeout:
            process.terminate()
            return False, "Execution timed out."

        if psutil.virtual_memory().percent > 90:
            process.terminate()
            return False, "Memory usage exceeded limit."

        time.sleep(0.1)

    process.join()

    if 'error' in return_dict and return_dict['error']:
        return False, f"Error:\n{return_dict['error']}"

    if 'output' in return_dict and return_dict['output']:
        st.text("Print output:")
        st.code(return_dict['output'], language="")

    # Refresh and display files generated after execution
    display_generated_files()

    return True, "Code executed successfully."

# File system management functions
def save_file(filename, content):
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    return os.listdir("generated_files")

# Function to save various media outputs
def save_image(img, filename):
    img.save(f"generated_files/{filename}")
    display_generated_file(f"generated_files/{filename}")

def save_audio(audio, sample_rate, filename):
    sf.write(f"generated_files/{filename}", audio, sample_rate)
    display_generated_file(f"generated_files/{filename}")

# Function to display an individual file based on type
def display_generated_file(filepath):
    if filepath.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        st.image(filepath, use_column_width=True)
    elif filepath.endswith(('.wav', '.mp3')):
        audio_bytes = open(filepath, 'rb').read()
        st.audio(audio_bytes)
    elif filepath.endswith('.mp4'):
        st.video(filepath)
    elif filepath.endswith('.html'):
        with open(filepath, 'r') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600)
    else:
        # Display plain text or other non-media files
        with open(filepath, 'r') as f:
            content = f.read()
        st.text(content)

# Function to display all generated files with download options
# Display generated files only if they exist
def display_generated_files():
    files = list_files()
    if files:
        st.markdown("### Generated Files")
        for i, file in enumerate(files):
            filepath = f"generated_files/{file}"
            with st.expander(f"View {file}", expanded=False):
                display_generated_file(filepath)
            with open(filepath, "rb") as f:
                st.download_button(
                    label=f"Download {file}",
                    data=f,
                    file_name=file,
                    mime="application/octet-stream",
                    key=f"download_button_{i}"  # Unique key for each button
                )
    else:
        st.info("No generated files yet.")



def capture_widget_states(local_vars):
    widget_states = {}
    for name, value in local_vars.items():
        if name.startswith('st_'):
            if hasattr(value, 'value'):
                widget_states[name] = value.value
    return widget_states

def restore_widget_states(widget_states):
    for name, value in widget_states.items():
        if name in st.session_state:
            st.session_state[name] = value

def get_streamlit_widgets():
    return [name for name, func in inspect.getmembers(st) 
            if inspect.isfunction(func) and name.startswith('slider') or name.startswith('text_input') or name.startswith('checkbox')]

def get_code_suggestions(current_line):
    suggestions = []
    
    if 'st.' in current_line:
        widget_funcs = get_streamlit_widgets()
        suggestions.extend([func for func in widget_funcs if func.startswith(current_line.split('.')[-1])])
    
    return suggestions

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

# Function to convert Pygame surface to PIL Image
def pygame_surface_to_image(surface):
    buffer = surface.get_view("RGB")
    return Image.frombytes("RGB", surface.get_size(), buffer.raw)


# New functions to save various outputs
def save_plot(fig, filename):
    fig.savefig(f"generated_files/{filename}")

def save_plotly(fig, filename):
    fig.write_html(f"generated_files/{filename}")


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

def main():
    st.title("🚀 Advanced Coding Assistant")
    
    # Clear generated files on start
    clear_generated_files()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your coding assistant. Ask me anything or request code!"})
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'last_code' not in st.session_state:
        st.session_state.last_code = None
    if 'code_editor' not in st.session_state:
        st.session_state.code_editor = ""

    # Create directory for generated files if not exists
    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

    # Sidebar for API key input
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")

    # Display previous chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Code execution area
    st.markdown("### Code Execution Area")
    code_editor = st_ace(
        value=st.session_state.code_editor,
        language="python",
        theme="monokai",
        keybinding="vscode",
        show_gutter=True,
        show_print_margin=True,
        wrap=True,
        auto_update=True,
        font_size=14,
        tab_size=4,
        placeholder="Write your Python code here...",
        key="ace_editor"
    )

    # Update session state with current code
    st.session_state.code_editor = code_editor
    st.session_state.last_code = code_editor

    # Process chat input
    prompt = st.chat_input("Ask for code or any coding question...", key="chat_input")
    
    if prompt:
        # Process chat using GPT-4 and add to conversation
        if api_key:
            with st.spinner("Processing..."):
                response = chat_with_gpt(prompt, api_key, st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                # If response contains code, populate the editor with it
                if "```python" in response:
                    code_block = response.split("```python")[1].split("```")[0].strip()
                    st.session_state.code_editor = code_block
                    st.session_state.last_code = code_block
        else:
            st.warning("Please enter your OpenAI API key.")

    # Code execution buttons
    if st.button("🏃‍♂️ Run Code"):
        if st.session_state.last_code:
            with st.spinner("Executing code..."):
                success, message = execute_code(st.session_state.last_code)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.warning("Please write some code to execute.")

    if st.button("🔧 Fix and Rerun"):
        if st.session_state.last_error and st.session_state.last_code:
            with st.spinner("Fixing code..."):
                fixed_code = fix_code(st.session_state.last_code, st.session_state.last_error, api_key)
                st.session_state.code_editor = fixed_code
                st.session_state.last_code = fixed_code

    # Display generated files if any
    display_generated_files()

# Entry point
if __name__ == "__main__":
    main()

