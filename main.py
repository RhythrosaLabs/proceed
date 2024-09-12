# Supercharged Advanced Coding Assistant
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
import time

# Set page configuration
st.set_page_config(
    page_title="Supercharged Coding Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved UI with padding
st.markdown("""
    <style>
        /* General settings */
        body {
            background-color: #0e1117;
            color: #c9d1d9;
        }
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        /* Chat messages */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: row;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #161b22;
        }
        .chat-message.assistant {
            background-color: #21262d;
        }
        .chat-message .message {
            width: 100%;
            padding-left: 1rem;
        }
        /* Buttons */
        .stButton>button {
            background-color: #238636;
            color: white;
            border-radius: 5px;
            padding: 0.5rem;
            margin: 0.25rem 0;
            transition: all 0.2s ease-in-out;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #2ea043;
        }
        /* Input fields */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #161b22;
            color: #c9d1d9;
            border-radius: 5px;
        }
        /* Code execution area */
        .code-execution-area {
            background-color: #161b22;
            border-radius: 5px;
            padding: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #484f58;
            border-radius: 5px;
        }
        /* File explorer */
        .file-explorer {
            background-color: #21262d;
            padding: 1rem;
            border-radius: 5px;
        }
        /* Syntax highlighting */
        .ace_editor {
            background: #0e1117;
        }
        .ace_gutter {
            background: #0e1117;
            color: #6e7681;
        }
        .ace_keyword {
            color: #ff7b72;
        }
        .ace_string {
            color: #a5d6ff;
        }
        .ace_comment {
            color: #8b949e;
        }
        .ace_identifier {
            color: #d2a8ff;
        }
        /* Padding for the main content */
        .main-content {
            padding: 0 2rem;  /* Add padding on left and right */
            margin-bottom: 120px; /* Adjusted for the bottom bar */
        }
        /* Bottom bar styling */
        .bottom-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #0e1117;
            padding: 0.5rem 1rem;
            z-index: 9999;
            border-top: 1px solid #30363d;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .bottom-bar .stTextInput, .bottom-bar .stButton {
            margin-bottom: 0;
        }
        .bottom-bar .stTextInput>div>div {
            flex-grow: 1;
        }
        .bottom-bar .stButton>button {
            width: 100%;
            margin: 0;
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
        'list_files': list_files,
        'uploaded_file': st.session_state.get('uploaded_file', None),
    }
    try:
        exec(code, globals(), local_vars)
        st.success("Code executed successfully.")
    except Exception as e:
        st.error(f"Error executing code: {str(e)}")
        st.session_state.last_error = str(e)

# Functions for file system management
def save_file(filename, content):
    with open(f"generated_files/{filename}", "w") as f:
        f.write(content)

def load_file(filename):
    with open(f"generated_files/{filename}", "r") as f:
        return f.read()

def list_files():
    return os.listdir("generated_files")

# Function to call GPT-4 via requests
def chat_with_gpt(prompt, api_key, conversation_history):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = conversation_history + [{"role": "user", "content": prompt}]
    data = {
        "model": st.session_state.selected_model,
        "messages": messages,
        "max_tokens": 2500,
        "temperature": st.session_state.temperature,
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display chat messages
def display_chat_message(role, content):
    if role == "user":
        st.markdown(f'<div class="chat-message user"><div class="message">{content}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant"><div class="message">{content}</div></div>', unsafe_allow_html=True)

# Function to fix code
def fix_code_function(code, error_message, api_key):
    prompt = f"The following Python code produced an error:\n\n```python\n{code}\n```\n\nError message: {error_message}\n\nPlease provide a corrected version of the code that fixes this error."
    fixed_code = chat_with_gpt(prompt, api_key, [])
    return fixed_code

# Main function to run the Streamlit app
def main():
    st.title("🚀 Supercharged Coding Assistant")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your supercharged coding assistant. How can I help you today?"})
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'last_code' not in st.session_state:
        st.session_state.last_code = ""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4"
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7

    # Create directory for generated files if it doesn't exist
    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

    # Sidebar for settings and additional features
    with st.sidebar:
        st.header("🔑 Settings")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Your API key is needed to communicate with OpenAI's GPT-4 model.")
        st.selectbox("Select Model", ["gpt-4", "gpt-3.5-turbo"], key='selected_model')
        st.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.7, key='temperature', help="Controls the creativity of the AI's responses.")
        st.markdown("---")
        st.subheader("📁 File Management")
        uploaded_file = st.file_uploader("Upload a file", type=["py", "txt", "csv", "json", "jpg", "png", "wav"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            if not os.path.exists("generated_files"):
                os.makedirs("generated_files")
            with open(os.path.join("generated_files", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        st.markdown("---")
        st.subheader("📚 Available Libraries and Features:")
        st.markdown("""
        - **Plotting:** matplotlib, seaborn, plotly, altair
        - **Data:** pandas, numpy
        - **Geospatial:** pydeck
        - **Audio:** librosa, pedalboard, mido, soundfile, sox
        - **Image:** PIL, cv2
        - **Game Development:** pygame
        - **File System:** save_file, load_file, list_files
        - **Others:** io, base64
        """)
        st.markdown("---")
        st.caption("Note: Your API key is stored securely in the app session and not shared.")

    # Wrap main content in a container with padding
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Display chat messages
    st.markdown("### 💬 Chat History")
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Code execution area with code editor
    st.markdown("### 📝 Code Execution Area")
    with st.expander("View/Hide Code Execution Area", expanded=True):
        st.markdown('<div class="code-execution-area">', unsafe_allow_html=True)
        code_editor = st_ace(
            value=st.session_state.last_code,
            language='python',
            theme='pastel_on_dark',
            keybinding='vscode',
            font_size=14,
            tab_size=4,
            wrap=True,
            auto_update=True,
            readonly=False,
            key="ace_code_editor"
        )
        st.session_state.last_code = code_editor
        st.markdown('</div>', unsafe_allow_html=True)

    # Display generated files with file explorer
    st.markdown("### 📁 Generated Files")
    files = list_files()
    if files:
        st.markdown('<div class="file-explorer">', unsafe_allow_html=True)
        selected_file = st.selectbox("Select a file to view:", files)
        if selected_file:
            file_path = os.path.join("generated_files", selected_file)
            if selected_file.endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(file_path)
                st.image(image, caption=selected_file)
            elif selected_file.endswith(('.wav', '.mp3')):
                audio_file = open(file_path, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            else:
                content = load_file(selected_file)
                st.text_area(f"Contents of {selected_file}", content, height=200)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No generated files yet.")

    # Close the main content container
    st.markdown('</div>', unsafe_allow_html=True)

    # Create the bottom bar
    with st.container():
        st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)

        # Create columns for the input field and action buttons
        col_input, col_buttons = st.columns([3, 2])

        with col_input:
            prompt = st.text_input("✍️ Your Input: Ask me anything about coding or request a visualization...", key="user_input")

        with col_buttons:
            # Arrange buttons horizontally
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                run_code = st.button("🏃‍♂️ Run Code")
            with col2:
                fix_code = st.button("🔧 Fix Code")
            with col3:
                save_code = st.button("💾 Save Code")
            with col4:
                clear_code = st.button("🧹 Clear Code")
            with col5:
                clear_chat = st.button("🗑️ Clear Chat")

        st.markdown('</div>', unsafe_allow_html=True)

    # Update button action handlers
    if run_code:
        if st.session_state.last_code.strip() != "":
            with st.spinner("Executing code..."):
                execute_code(st.session_state.last_code)
        else:
            st.warning("No code to execute. Please write or request some code first.")

    if fix_code:
        if st.session_state.last_error and st.session_state.last_code.strip() != "":
            if openai_api_key:
                with st.spinner("Fixing code..."):
                    fixed_code = fix_code_function(st.session_state.last_code, st.session_state.last_error, openai_api_key)
                    st.session_state.last_code = fixed_code
                    st.success("Code has been fixed. Please rerun.")
            else:
                st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            st.warning("No error to fix or no previous code execution. Please run some code first.")

    if save_code:
        if st.session_state.last_code.strip() != "":
            with st.spinner("Saving code..."):
                save_file_name = st.text_input("Enter filename to save code:", value="my_code.py")
                if save_file_name:
                    save_file(save_file_name, st.session_state.last_code)
                    st.success(f"Code saved as '{save_file_name}' in 'generated_files' directory.")
        else:
            st.warning("No code to save.")

    if clear_code:
        st.session_state.last_code = ""
        st.session_state.last_error = None

    if clear_chat:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Chat cleared. How can I assist you?"})

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

# Entry point
if __name__ == "__main__":
    main()
