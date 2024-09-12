# Optimized Advanced Coding Assistant
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

# Set page configuration
st.set_page_config(
    page_title="Advanced Coding Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved UI
st.markdown("""
    <style>
        /* General settings */
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Chat messages */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #2e2e2e;
        }
        .chat-message.assistant {
            background-color: #3e3e3e;
        }
        .chat-message .avatar {
            width: 10%;
        }
        .chat-message .avatar img {
            max-width: 50px;
            max-height: 50px;
            border-radius: 50%;
            object-fit: cover;
        }
        .chat-message .message {
            width: 90%;
            padding-left: 1rem;
        }
        /* Buttons */
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            margin-right: 0.5rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #5753c9;
        }
        /* Input fields */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #2e2e2e;
            color: white;
            border-radius: 10px;
        }
        /* Code execution area */
        .code-execution-area {
            background-color: #2e2e2e;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #5753c9;
            border-radius: 10px;
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
    if 'plt' in local_vars and plt.get_fignums():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()
    return local_vars

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
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 2500
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
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your advanced coding assistant. How can I help you today?"})
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'last_code' not in st.session_state:
        st.session_state.last_code = None

    # Create directory for generated files if it doesn't exist
    if not os.path.exists("generated_files"):
        os.makedirs("generated_files")

    # Sidebar for API key input and library information
    with st.sidebar:
        st.header("🔑 Settings")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Your API key is needed to communicate with OpenAI's GPT-4 model.")
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

    # Display chat messages
    st.markdown("### 💬 Chat History")
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Code execution area
    st.markdown("### 📝 Code Execution Area")
    with st.expander("View/Hide Code Execution Area", expanded=True):
        st.markdown('<div class="code-execution-area">', unsafe_allow_html=True)
        if st.session_state.last_code:
            st.code(st.session_state.last_code, language="python")
        else:
            st.info("No code to display. Request a code sample or write some code to get started!")
            example_code = """
# Example: Create an interactive scatter plot with Plotly
import plotly.express as px
import numpy as np
import pandas as pd

# Generate some random data
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'size': np.random.randint(1, 20, 100)
})

# Create a scatter plot
fig = px.scatter(data, x='x', y='y', size='size', color='size',
                 title='Interactive Scatter Plot')
fig.update_layout(template='plotly_dark')

# Display the plot
st.plotly_chart(fig)

# Save the data
data.to_csv('generated_files/scatter_data.csv', index=False)
st.write("Data saved to 'scatter_data.csv'")
"""
            st.code(example_code, language="python")
            if st.button("Try This Example"):
                st.session_state.last_code = example_code
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    st.markdown("### ✍️ Your Input")
    prompt = st.text_input("Ask me anything about coding or request a visualization...")

    # Action buttons
    st.markdown("### 🔧 Actions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏃‍♂️ Run Code"):
            if st.session_state.last_code:
                with st.spinner("Executing code..."):
                    try:
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
        if st.button("🔧 Fix Code"):
            if st.session_state.last_error and st.session_state.last_code:
                if openai_api_key:
                    with st.spinner("Fixing code..."):
                        fixed_code = fix_code(st.session_state.last_code, st.session_state.last_error, openai_api_key)
                        st.session_state.last_code = fixed_code
                        st.success("Code has been fixed. Please rerun.")
                else:
                    st.warning("Please enter your OpenAI API key in the sidebar.")
            else:
                st.warning("No error to fix or no previous code execution. Please run some code first.")
    with col3:
        if st.button("🧹 Clear Code"):
            st.session_state.last_code = None
            st.session_state.last_error = None
    with col4:
        if st.button("🗑️ Clear Chat"):
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

    # Display generated files
    st.markdown("### 📁 Generated Files")
    files = list_files()
    if files:
        for file in files:
            if st.button(f"View {file}"):
                content = load_file(file)
                st.text_area(f"Contents of {file}", content, height=200)
    else:
        st.info("No generated files yet.")

# Entry point
if __name__ == "__main__":
    main()
