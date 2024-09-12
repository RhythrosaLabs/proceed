import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pydeck as pdk
import librosa
import cv2
from PIL import Image
import io
import base64
import soundfile as sf
import os
import contextlib
import plotly.graph_objs as go
import multiprocessing
import time
import psutil
import traceback
import json
import re
import uuid

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Advanced Coding Assistant", page_icon="🚀")

# Custom CSS for a more appealing UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
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
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    .code-editor {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .file-display {
        border: 1px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: rgba(0, 0, 0, 0.2);
    }
    .sidebar .stButton>button {
        width: 100%;
    }
    .hljs {
        background: transparent !important;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def ensure_directory_exists():
    os.makedirs("generated_files", exist_ok=True)

def clear_generated_files():
    for filename in os.listdir("generated_files"):
        file_path = os.path.join("generated_files", filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

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

def save_image(img, filename):
    ensure_directory_exists()
    img.save(f"generated_files/{filename}")

def save_audio(audio, sample_rate, filename):
    ensure_directory_exists()
    sf.write(f"generated_files/{filename}", audio, sample_rate)

def save_plot(fig, filename):
    ensure_directory_exists()
    fig.savefig(f"generated_files/{filename}")

def save_plotly(fig, filename):
    ensure_directory_exists()
    fig.write_html(f"generated_files/{filename}")

def display_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension in ['.png', '.jpg', '.jpeg', '.gif']:
        st.image(file_path, use_column_width=True)
    elif file_extension in ['.wav', '.mp3']:
        st.audio(file_path)
    elif file_extension == '.mp4':
        st.video(file_path)
    elif file_extension == '.html':
        with open(file_path, 'r') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600)
    else:
        with open(file_path, 'r') as f:
            content = f.read()
        st.code(content, language='python')

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
            'save_plot': save_plot,
            'save_plotly': save_plotly,
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

    return True, "Code executed successfully."

def chat_with_gpt(prompt, api_key, conversation_history):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = conversation_history + [{"role": "user", "content": prompt}]
    data = {
        "model": "gpt-4",
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

def fix_code(code, error_message, api_key):
    prompt = f"The following Python code produced an error:\n\n```python\n{code}\n```\n\nError message: {error_message}\n\nPlease provide a corrected version of the code that fixes this error."
    fixed_code = chat_with_gpt(prompt, api_key, [])
    return fixed_code

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

def save_session():
    session_data = {
        "code": st.session_state.code_editor,
        "chat": st.session_state.messages,
        "history": st.session_state.code_history
    }
    with open("session.json", "w") as f:
        json.dump(session_data, f)
    st.success("Session saved successfully!")

def load_session():
    try:
        with open("session.json", "r") as f:
            session_data = json.load(f)
        st.session_state.code_editor = session_data["code"]
        st.session_state.messages = session_data["chat"]
        st.session_state.code_history = session_data["history"]
        st.success("Session loaded successfully!")
    except FileNotFoundError:
        st.error("No saved session found.")

def add_to_history(code):
    if code not in st.session_state.code_history:
        st.session_state.code_history.append(code)
    if len(st.session_state.code_history) > 10:
        st.session_state.code_history.pop(0)

def main():
    st.title("🚀 Advanced Coding Assistant v3")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your advanced coding assistant. How can I help you today?"}]
    if 'code_editor' not in st.session_state:
        st.session_state.code_editor = ""
    if 'code_history' not in st.session_state:
        st.session_state.code_history = []
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")
        
        st.subheader("Session Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Session", key="save_session"):
                save_session()
        with col2:
            if st.button("Load Session", key="load_session"):
                load_session()
        
        st.subheader("Code History")
        for i, historical_code in enumerate(st.session_state.code_history):
            if st.button(f"Load #{i+1}", key=f"history_{i}"):
                st.session_state.code_editor = historical_code
                st.experimental_rerun()

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat with AI")
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])

        prompt = st.chat_input("Ask for code or any coding question...", key="chat_input")
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)

            if api_key:
                with st.spinner("AI is thinking..."):
                    response = chat_with_gpt(prompt, api_key, st.session_state.messages[:-1])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    display_chat_message("assistant", response)
                    
                    if "```python" in response:
                        code_block = response.split("```python")[1].split("```")[0].strip()
                        st.session_state.code_editor = code_block
                        add_to_history(code_block)
                        st.experimental_rerun()
            else:
                st.warning("Please enter your OpenAI API key in the sidebar.")

    with col2:
        st.subheader("🖥️ Code Executor")
        
        st.session_state.code_editor = st.text_area("Python Code", value=st.session_state.code_editor, height=300, key="code_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏃‍♂️ Run Code", key="run_code"):
                if st.session_state.code_editor:
                    with st.spinner("Executing code..."):
                        success, message = execute_code(st.session_state.code_editor)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                            st.session_state.last_error = message
                else:
                    st.warning("Please write some code to execute.")

        with col2:
            if st.button("🔧 Fix and Rerun", key="fix_and_rerun"):
                if st.session_state.last_error and st.session_state.code_editor:
                    with st.spinner("Fixing code..."):
                        fixed_code = fix_code(st.session_state.code_editor, st.session_state.last_error, api_key)
                        st.session_state.code_editor = fixed_code
                        add_to_history(fixed_code)
                        success, message = execute_code(fixed_code)
                        if success:
                            st.success("Code fixed and executed successfully.")
                            st.session_state.last_error = None
                        else:
                            st.error(f"Error after fixing: {message}")
                            st.session_state.last_error = message
                else:
                    st.warning("No error to fix or no previous code execution.")

        if st.button("🧹 Clear Code", key="clear_code"):
            st.session_state.code_editor = ""
            st.session_state.last_error = None
            st.experimental_rerun()

    # Display generated files
    st.subheader("📁 Generated Files")
    files = list_files()
    if files:
        for file in files:
            with st.expander(f"{file}", expanded=False):
                file_path = os.path.join("generated_files", file)
                display_file(file_path)
                
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"Download {file}",
                        data=f,
                        file_name=file,
                        mime="application/octet-stream",
                        key=f"download_{uuid.uuid4()}"
                    )
    else:
        st.info("No generated files yet. Run some code that generates output to see files here.")

    # Add a footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Advanced Coding Assistant")

# Entry point
if __name__ == "__main__":
    main()
