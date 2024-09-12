import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from streamlit_ace import st_ace
import json
import os
import io
import base64
import soundfile as sf
import multiprocessing
import contextlib
import time
import psutil

# Simplified Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 20px; }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.05); }
    .stTextInput>div>div>input { background-color: rgba(255, 255, 255, 0.1); color: white; border-radius: 10px; }
    .file-display, .code-block { background-color: rgba(49, 51, 63, 0.9); color: white; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None

# Ensure generated_files folder exists
def ensure_directory_exists():
    os.makedirs("generated_files", exist_ok=True)

# Clear all generated files
def clear_generated_files():
    folder = "generated_files"
    ensure_directory_exists()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

# File system management functions
def save_file(filename, content):
    ensure_directory_exists()
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    return os.listdir("generated_files")

# Save & display different types of generated files
def save_image(img, filename):
    ensure_directory_exists()
    img.save(f"generated_files/{filename}")
    display_generated_file(f"generated_files/{filename}")

def save_audio(audio, sample_rate, filename):
    ensure_directory_exists()
    sf.write(f"generated_files/{filename}", audio, sample_rate)
    display_generated_file(f"generated_files/{filename}")

def display_generated_file(filepath):
    if filepath.endswith(('.png', '.jpg', '.jpeg')):
        st.image(filepath, use_column_width=True)
    elif filepath.endswith(('.wav', '.mp3')):
        st.audio(filepath)
    elif filepath.endswith('.html'):
        with open(filepath, 'r') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600)
    else:
        st.text(load_file(filepath))

# Execute code with resource and time limits
def execute_code(code, timeout=30):
    def worker(code, return_dict):
        local_vars = {
            'st': st, 'save_file': save_file, 'load_file': load_file,
            'list_files': list_files, 'save_image': save_image,
            'save_audio': save_audio, 'display_generated_file': display_generated_file
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

    if 'error' in return_dict and return_dict['error']:
        return False, f"Error:\n{return_dict['error']}"
    
    st.text("Print output:")
    st.code(return_dict['output'])
    display_generated_files()
    return True, "Code executed successfully."

# Display generated files with download options
def display_generated_files():
    files = list_files()
    if files:
        st.markdown("### Generated Files")
        for file in files:
            filepath = f"generated_files/{file}"
            display_generated_file(filepath)
            with open(filepath, "rb") as f:
                st.download_button(f"Download {file}", f, file_name=file)

# Main function to run the app
def main():
    st.title("🚀 Advanced Coding Assistant")
    
    # Clear previous files
    clear_generated_files()

    # Initialize session state
    if 'code_editor' not in st.session_state:
        st.session_state.code_editor = ""

    # Sidebar for API key input
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")

    # Code editor
    code_editor = st_ace(
        value=st.session_state.code_editor,
        language="python",
        theme="monokai",
        key="ace_editor"
    )
    st.session_state.code_editor = code_editor

    # Execute code button
    if st.button("🏃‍♂️ Run Code"):
        if code_editor:
            with st.spinner("Executing code..."):
                success, message = execute_code(code_editor)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Display generated files
    display_generated_files()

if __name__ == "__main__":
    main()
