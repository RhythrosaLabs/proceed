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
    .split-screen {
        display: flex;
        flex-direction: row;
        height: calc(100vh - 100px); /* Leave room for bottom input bar */
    }
    .code-runner, .main-chat {
        width: 50%;
        padding: 20px;
        overflow-y: auto;
    }
    .bottom-bar {
        position: fixed;
        bottom: 0;
        width: 100%;
        display: flex;
        justify-content: space-evenly;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        z-index: 10;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# Function to load Lottie animation (unchanged)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Updated function to execute user-provided code
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
        'save_file': save_file,
        'load_file': load_file,
        'list_files': list_files
    }
    
    output = io.StringIO()
    
    try:
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
        
        if 'image' in local_vars and isinstance(local_vars['image'], Image.Image):
            st.image(local_vars['image'], caption="Generated Image", use_column_width=True)
            save_image(local_vars['image'], "generated_image.png")
        
        return True, "Code executed successfully."
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

    # Sidebar for API key input
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        st.markdown("---")
        st.markdown("### Quick Tips:")
        st.markdown("1. Chat naturally about coding tasks")
        st.markdown("2. Request code samples for various visualizations")
        st.markdown("3. Experiment with image and data processing")
        st.markdown("4. Use the buttons at the bottom to manage your code and chat")
        
        st.markdown("---")
        st.markdown("### Available Libraries:")
        st.markdown("- Plotting: matplotlib, seaborn, plotly, altair")
        st.markdown("- Data: pandas, numpy")
        st.markdown("- Geospatial: pydeck")
        st.markdown("- Audio: librosa")
        st.markdown("- Image: PIL, cv2")
        st.markdown("- Others: io, base64")

    # Split screen layout
    col1, col2 = st.columns(2)

    with col1:
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
# Example: Create a scatter plot with Plotly and save it
import plotly.express as px
import numpy as np

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

# Save the plot
save_plotly(fig, "interactive_scatter_plot.html")
st.write("Plot saved as 'interactive_scatter_plot.html'")

# Save the data
data.to_csv('generated_files/scatter_data.csv', index=False)
st.write("Data saved as 'scatter_data.csv'")
"""
                st.code(example_code, language="python")
                if st.button("Try This Example"):
                    st.session_state.last_code = example_code
                    st.experimental_rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # Run Code and Fix and Rerun buttons
        col1_1, col1_2 = st.columns(2)
        with col1_1:
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

        with col1_2:
            if st.button("🔧 Fix and Rerun", key="fix_and_rerun"):
                if st.session_state.last_error and st.session_state.last_code:
                    with st.spinner("Fixing code..."):
                        fixed_code = fix_code(st.session_state.last_code, st.session_state.last_error, api_key)
                        st.session_state.last_code = fixed_code
                        success, message = execute_code(fixed_code)
                        if success:
                            st.success("Code fixed and executed successfully.")
                            st.session_state.last_error = None
                        else:
                            st.error(f"Error after fixing: {message}")
                            st.session_state.last_error = message
                else:
                    st.warning("No error to fix or no previous code execution. Please run some code first.")

        # Display generated files
        st.markdown("### Generated Files")
        files = list_files()
        if files:
            for file in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        st.image(f"generated_files/{file}", caption=file, use_column_width=True)
                    elif file.endswith('.html'):
                        with open(f"generated_files/{file}", 'r') as f:
                            html_string = f.read()
                        st.components.v1.html(html_string, height=600)
                    else:
                        content = load_file(file)
                        st.text_area(f"Content of {file}", content, height=200)
                with col2:
                    with open(f"generated_files/{file}", "rb") as f:
                        st.download_button(
                            label=f"Download {file}",
                            data=f,
                            file_name=file,
                            mime="application/octet-stream"
                        )

    with col2:
        st.markdown("### Chat Interface")
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about coding or request a visualization..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in chat_with_gpt(prompt, api_key, st.session_state.messages[:-1]):
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Update last_code if the response contains a code block
            if "```python" in full_response:
                st.session_state.last_code = full_response.split("```python")[1].split("```")[0].strip()
                st.experimental_rerun()

    # Bottom action bar with input and buttons
    st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🧹 Clear Code", key="clear_code"):
            st.session_state.last_code = None
            st.session_state.last_error = None
            st.experimental_rerun()

    with col2:
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Chat cleared. How can I assist you?"})
            st.experimental_rerun()

    with col3:
        if st.button("🔄 Reset All", key="reset_all"):
            st.session_state.messages = []
            st.session_state.last_error = None
            st.session_state.last_code = None
            st.session_state.messages.append({"role": "assistant", "content": "Everything has been reset. How can I help you today?"})
            st.experimental_rerun()

    with col4:
        if st.button("💾 Save Session", key="save_session"):
            # Implement session saving functionality here
            st.success("Session saved successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

# Entry point
if __name__ == "__main__":
    main()
