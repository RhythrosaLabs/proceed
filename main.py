import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_ace import st_ace
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
import inspect

# Custom CSS with improved styling
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
    .code-block {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
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
    .stChatInputContainer > div {
        margin-bottom: 0 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def ensure_directory_exists():
    os.makedirs("generated_files", exist_ok=True)

def clear_generated_files():
    for filename in os.listdir("generated_files"):
        file_path = os.path.join("generated_files", filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

# File management functions
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
    display_generated_file(f"generated_files/{filename}")

def save_audio(audio, sample_rate, filename):
    ensure_directory_exists()
    sf.write(f"generated_files/{filename}", audio, sample_rate)
    display_generated_file(f"generated_files/{filename}")

def save_plot(fig, filename):
    ensure_directory_exists()
    fig.savefig(f"generated_files/{filename}")

def save_plotly(fig, filename):
    ensure_directory_exists()
    fig.write_html(f"generated_files/{filename}")

# Display functions
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

# Code execution function
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

    display_generated_files()

    return True, "Code executed successfully."

# GPT-4 interaction function
def chat_with_gpt(prompt, api_key, conversation_history):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = conversation_history + [{"role": "user", "content": prompt}]
    data = {
        "model": "gpt-4o",
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

# Code fixing function
def fix_code(code, error_message, api_key):
    prompt = f"The following Python code produced an error:\n\n```python\n{code}\n```\n\nError message: {error_message}\n\nPlease provide a corrected version of the code that fixes this error."
    fixed_code = chat_with_gpt(prompt, api_key, [])
    return fixed_code

# Main function
# New function for session management
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

# New function for code history
def add_to_history(code):
    if 'code_history' not in st.session_state:
        st.session_state.code_history = []
    if code not in st.session_state.code_history:
        st.session_state.code_history.append(code)
    if len(st.session_state.code_history) > 10:  # Keep only the last 10 entries
        st.session_state.code_history.pop(0)

# Main function
def main():
    st.title("🚀 Advanced Coding Assistant v2")
    
    clear_generated_files()

    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your coding assistant. Ask me anything or request code!"}]
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'code_editor' not in st.session_state:
        st.session_state.code_editor = ""
    if 'code_history' not in st.session_state:
        st.session_state.code_history = []

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")
        
        st.subheader("Session Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Session"):
                save_session()
        with col2:
            if st.button("Load Session"):
                load_session()
        
        st.subheader("Code History")
        for i, historical_code in enumerate(st.session_state.code_history):
            if st.button(f"Load #{i+1}", key=f"history_{i}"):
                st.session_state.code_editor = historical_code
                st.experimental_rerun()

    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    st.markdown("### Enhanced Code Editor")
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
        show_line_numbers=True,
        key="ace_editor"
    )

    if code_editor != st.session_state.code_editor:
        st.session_state.code_editor = code_editor
        add_to_history(code_editor)

    prompt = st.chat_input("Ask for code or any coding question...", key="chat_input")
    
    if prompt:
        if api_key:
            with st.spinner("Processing..."):
                response = chat_with_gpt(prompt, api_key, st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                if "```python" in response:
                    code_block = response.split("```python")[1].split("```")[0].strip()
                    st.session_state.code_editor = code_block
                    add_to_history(code_block)
        else:
            st.warning("Please enter your OpenAI API key.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🏃‍♂️ Run Code"):
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
        if st.button("🔧 Fix and Rerun"):
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
                st.warning("No error to fix or no previous code execution. Please run some code first.")

    with col3:
        if st.button("🧹 Clear Code"):
            st.session_state.code_editor = ""
            st.session_state.last_error = None
            st.experimental_rerun()

    with col4:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. How can I assist you?"}]
            st.experimental_rerun()

    with col5:
        if st.button("🔄 Reset All"):
            st.session_state.messages = [{"role": "assistant", "content": "Everything has been reset. How can I help you today?"}]
            st.session_state.last_error = None
            st.session_state.code_editor = ""
            st.session_state.code_history = []
            clear_generated_files()
            st.experimental_rerun()

    # Display generated files
    display_generated_files()

    # Add a section for code suggestions and auto-completion
    st.markdown("### Code Suggestions")
    if st.checkbox("Enable Code Suggestions"):
        current_line = st.text_input("Current line of code:")
        suggestions = get_code_suggestions(current_line)
        if suggestions:
            st.write("Suggestions:")
            for suggestion in suggestions:
                if st.button(suggestion):
                    st.session_state.code_editor += suggestion
                    st.experimental_rerun()

# Function to get code suggestions
def get_code_suggestions(current_line):
    suggestions = []
    
    if 'st.' in current_line:
        widget_funcs = get_streamlit_widgets()
        suggestions.extend([func for func in widget_funcs if func.startswith(current_line.split('.')[-1])])
    
    # Add more suggestion logic here for other libraries
    
    return suggestions

def get_streamlit_widgets():
    return [name for name, func in inspect.getmembers(st) 
            if inspect.isfunction(func) and (name.startswith('slider') or name.startswith('text_input') or name.startswith('checkbox'))]

# Entry point
if __name__ == "__main__":
    main()
