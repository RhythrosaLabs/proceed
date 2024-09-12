import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
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

# Custom CSS for displaying the files on the right side
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
    .file-display {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.9);
        color: white;
    }
    .file-display h6 {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    /* Align file display on the right */
    .block-container {
        display: flex;
        justify-content: space-between;
    }
    .left-content {
        width: 65%;
    }
    .right-content {
        width: 30%;
        background-color: rgba(49, 51, 63, 0.9);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to ensure directory exists
def ensure_directory_exists():
    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

# Function to clear all generated files
def clear_generated_files():
    ensure_directory_exists()
    folder = "generated_files"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

# File system management functions
def save_file(filename, content):
    ensure_directory_exists()
    file_path = os.path.join("generated_files", filename)
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

def load_file(filename):
    ensure_directory_exists()
    file_path = os.path.join("generated_files", filename)
    with open(file_path, "r") as f:
        return f.read()

def list_files():
    ensure_directory_exists()
    return os.listdir("generated_files")

# Function to save and display images
def save_image(img, filename):
    ensure_directory_exists()
    img.save(f"generated_files/{filename}")
    return f"generated_files/{filename}"

# Function to save and display audio
def save_audio(audio, sample_rate, filename):
    ensure_directory_exists()
    sf.write(f"generated_files/{filename}", audio, sample_rate)
    return f"generated_files/{filename}"

# Display a file based on type
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
        with open(filepath, 'r') as f:
            content = f.read()
        st.text(content)

# Function to display all generated files
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
                    key=f"download_button_{i}"
                )
    else:
        st.info("No generated files yet.")

# Function to execute user-provided code
def execute_code(code, timeout=30):
    def worker(code, return_dict):
        local_vars = {
            'st': st,
            'save_file': save_file,
            'load_file': load_file,
            'list_files': list_files,
            'save_image': save_image,
            'save_audio': save_audio,
            'display_generated_file': display_generated_file,
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
            'mido': mido,
            'pygame': pygame,
            'sf': sf,
            'sox': sox,
            'go': go
        }

        output = io.StringIO()
        error_output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
                exec(code, globals(), local_vars)

            return_dict['output'] = output.getvalue()
            return_dict['error'] = error_output.getvalue()
        except Exception as e:
            return_dict['error'] = str(e)

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

    if return_dict.get('error'):
        return False, f"Error:\n{return_dict['error']}"
    else:
        if return_dict.get('output'):
            st.text("Print output:")
            st.code(return_dict['output'], language="")

        display_generated_files()

        return True, "Code executed successfully."

# Layout adjustment for displaying code on the left and files on the right
def main():
    st.title("🚀 Advanced Coding Assistant")

    # Create two-column layout
    left, right = st.columns([3, 1])
    
    with left:
        # Code editor
        st.markdown("### Code Execution Area")
        code_editor = st_ace(
            value=st.session_state.get('code_editor', ""),
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
        st.session_state['code_editor'] = code_editor

        # Run code button
        if st.button("🏃‍♂️ Run Code"):
            if code_editor:
                with st.spinner("Executing code..."):
                    success, message = execute_code(code_editor)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("No code to execute.")
    
    with right:
        # Display generated files
        st.markdown("### Generated Files")
        display_generated_files()

# Entry point
if __name__ == "__main__":
    main()
