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
from streamlit_player import st_player
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from typing import List, Tuple
from streamlit_d3graph import d3graph
import networkx as nx
from textblob import TextBlob
from wordcloud import WordCloud
import spacy
from spacy import displacy
import yfinance as yf
from datetime import datetime, timedelta
from pytube import YouTube

# Set page config for wide layout with a custom theme
st.set_page_config(
    page_title="🚀 Ultimate Code Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Ultimate Code Assistant\nThis is an advanced Streamlit application for coding assistance and data visualization."
    }
)

# Custom CSS for an even more sleek and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e94560;
    }
    
    .stButton>button {
        background-color: #e94560;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        background-color: #c73e54;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e94560;
        border-radius: 10px;
        border: 1px solid #e94560;
    }
    
    .file-display {
        border: 1px solid #e94560;
        padding: 15px;
        margin: 10px 0;
        border-radius: 15px;
        background-color: rgba(25, 25, 25, 0.5);
        transition: all 0.3s ease;
    }
    
    .file-display:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(233, 69, 96, 0.2);
    }
    
    .code-execution-area {
        background-color: rgba(25, 25, 25, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid #e94560;
    }
    
    .code-execution-area pre {
        margin-bottom: 0;
        color: #61dafb;
    }
    
    .bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: rgba(25, 25, 25, 0.9);
        padding: 15px;
        z-index: 999;
        display: flex;
        justify-content: space-around;
        backdrop-filter: blur(10px);
    }
    
    .bottom-bar .stButton > button {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    .main .block-container {
        padding-bottom: 6rem;
    }
    
    .stChatInputContainer {
        position: fixed;
        bottom: 5rem;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: rgba(25, 25, 25, 0.9);
        z-index: 998;
        backdrop-filter: blur(10px);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom styles for new components */
    .chart-wrapper {
        background-color: rgba(25, 25, 25, 0.5);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e94560;
    }
    
    .audio-visualizer {
        background-color: rgba(25, 25, 25, 0.5);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #61dafb;
    }
    
    .webcam-feed {
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid #e94560;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions (enhanced)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def ensure_directory_exists(directory="generated_files"):
    os.makedirs(directory, exist_ok=True)

def save_file(filename, content, directory="generated_files"):
    ensure_directory_exists(directory)
    with open(f"{directory}/{filename}", "w") as f:
        f.write(content)

def load_file(filename, directory="generated_files"):
    with open(f"{directory}/{filename}", "r") as f:
        return f.read()

def list_files(directory="generated_files"):
    ensure_directory_exists(directory)
    return os.listdir(directory)

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

# Enhanced code execution function
def execute_code(code):
    local_vars = {
        'st': st, 'px': px, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'alt': alt,
        'pdk': pdk, 'librosa': librosa, 'cv2': cv2, 'Image': Image, 'go': go,
        'WordCloud': WordCloud, 'TextBlob': TextBlob, 'spacy': spacy, 'displacy': displacy,
        'yf': yf, 'datetime': datetime, 'timedelta': timedelta, 'YouTube': YouTube
    }
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals(), local_vars)
        st.code(output.getvalue(), language="")
        
        # Handle various output types
        if 'fig' in local_vars:
            if isinstance(local_vars['fig'], go.Figure):
                st.plotly_chart(local_vars['fig'], use_container_width=True)
            elif isinstance(local_vars['fig'], plt.Figure):
                st.pyplot(local_vars['fig'])
        
        if 'chart' in local_vars and isinstance(local_vars['chart'], alt.Chart):
            st.altair_chart(local_vars['chart'], use_container_width=True)
        
        if 'deck' in local_vars and isinstance(local_vars['deck'], pdk.Deck):
            st.pydeck_chart(local_vars['deck'])
        
        if 'wordcloud' in local_vars and isinstance(local_vars['wordcloud'], WordCloud):
            st.image(local_vars['wordcloud'].to_image())
        
        plt.close('all')  # Close all matplotlib figures
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Enhanced chat function with GPT-4
def chat_with_gpt(prompt, api_key, conversation_history):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4",
        "messages": conversation_history + [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display chat messages with enhanced styling
def display_chat_message(role, content):
    with st.chat_message(role):
        st.markdown(f"<div style='border-left: 3px solid {'#e94560' if role == 'user' else '#61dafb'}; padding-left: 10px;'>{content}</div>", unsafe_allow_html=True)

# New function for audio visualization
def visualize_audio(audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 8))
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Spectrogram')
    st.pyplot(fig)

# New function for real-time video processing
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 200)
        return img_edges

# New function for creating interactive network graphs
def create_network_graph():
    nodes = [
        {"id": "A", "label": "Node A", "size": 20},
        {"id": "B", "label": "Node B", "size": 15},
        {"id": "C", "label": "Node C", "size": 25},
    ]
    edges = [
        {"source": "A", "target": "B", "type": "KNOWS"},
        {"source": "B", "target": "C", "type": "LIKES"},
        {"source": "C", "target": "A", "type": "WORKS_WITH"},
    ]
    config = {
        "nodeHighlightBehavior": True,
        "highlightColor": "#F7A7A6",
        "directed": True,
        "collapsible": True,
    }
    return d3graph(nodes=nodes, links=edges, config=config)

# Main function to run the enhanced app
def main():
    st.title("🚀 Ultimate Coding Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Ultimate Coding Assistant! How can I supercharge your coding experience today?"}]
    if 'code_editor' not in st.session_state:
        st.session_state.code_editor = ""
    
    # Sidebar with enhanced settings and features
    with st.sidebar:
        st.header("🛠️ Control Panel")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        st.subheader("🎨 Theme Settings")
        theme_color = st.color_picker("Pick a theme color", "#e94560")
        st.markdown(f"<style>.stApp {{ background: linear-gradient(135deg, {theme_color}22 0%, {theme_color}44 50%, {theme_color}66 100%); }}</style>", unsafe_allow_html=True)
        
        st.subheader("🧠 AI Model Settings")
        model_temperature = st.slider("AI Creativity", 0.0, 1.0, 0.7)
        
        st.subheader("📊 Data Visualization")
        chart_type = st.selectbox("Default Chart Type", ["Line", "Bar", "Scatter", "Area"])
        
        st.subheader("🎵 Audio Settings")
        audio_effects = st.multiselect("Audio Effects", ["Reverb", "Chorus", "Distortion"])
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Enhanced chat input with natural language processing
    prompt = st.chat_input("Ask for code, visualizations, or any coding wizardry...")
    if prompt:
        if api_key:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Perform sentiment analysis on the prompt
            blob = TextBlob(prompt)
            sentiment = blob.sentiment.polarity
            
            if sentiment > 0.5:
                st.balloons()
            elif sentiment < -0.5:
                st.error("I sense some frustration. Let me try my best to help you!")
            
            response = chat_with_gpt(prompt, api_key, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Extract code blocks and update code editor
            if "```python" in response:
                code_blocks = response.split("```python")
                code_block = code_blocks[1].split("```")[0].strip()
                st.session_state.code_editor = code_block
            
            # Perform named entity recognition on the response
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(response)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.sidebar.subheader("🔍 Named Entities")
                st.sidebar.write(entities)
            
            st.experimental_rerun()
        else:
            st.warning("Please enter an OpenAI API key in the sidebar.")
    
    # Enhanced Code Editor Section
    st.subheader("🖥️ Advanced Code Editor & Executor")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.code_editor = st_ace(
            value=st.session_state.code_editor,
            language="python",
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            tab_size=4,
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            auto_update=True,
            readonly=False,
            min_lines=20,
            key="ace_editor"
        )
    
    with col2:
        st.subheader("🛠️ Code Tools")
        if st.button("🏃‍♂️ Run Code"):
            with st.spinner("Executing code..."):
                execute_code(st.session_state.code_editor)
        
        if st.button("🧹 Format Code"):
            import autopep8
            formatted_code = autopep8.fix_code(st.session_state.code_editor)
            st.session_state.code_editor = formatted_code
            st.success("Code formatted successfully!")
        
        if st.button("📊 Visualize Code"):
            import ast
            import networkx as nx
            
            try:
                tree = ast.parse(st.session_state.code_editor)
                G = nx.Graph()
                
                for node in ast.walk(tree):
                    G.add_node(node.__class__.__name__)
                    for child in ast.iter_child_nodes(node):
                        G.add_edge(node.__class__.__name__, child.__class__.__name__)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold', ax=ax)
                st.pyplot(fig)
            except SyntaxError as e:
                st.error(f"Syntax error in code: {str(e)}")
    
    # Advanced Data Visualization Section
    st.subheader("📊 Advanced Data Visualization")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("📈 Stock Market Visualizer")
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL)", "AAPL")
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End date", datetime.now())
        
        if st.button("Fetch Stock Data"):
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Market Data'
            ))
            fig.update_layout(title=f'{ticker} Stock Price', yaxis_title='Stock Price (USD)')
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        st.subheader("🗺️ Interactive Map")
        map_data = pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon']
        )
        st.map(map_data)
    
    # Audio Processing Section
    st.subheader("🎵 Audio Processing & Visualization")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("Visualize Audio"):
            with st.spinner("Generating audio visualization..."):
                visualize_audio(audio_file)
        
        if st.button("Apply Audio Effects"):
            y, sr = librosa.load(audio_file)
            board = Pedalboard([Chorus(), Reverb(room_size=0.8)])
            effected = board(y, sr)
            
            # Save the effected audio
            sf.write("effected_audio.wav", effected, sr)
            st.audio("effected_audio.wav")
    
    # Real-time Video Processing
    st.subheader("📹 Real-time Video Processing")
    webrtc_ctx = webrtc_streamer(
        key="sample",
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )
    
    # Interactive Network Graph
    st.subheader("🕸️ Interactive Network Graph")
    create_network_graph()
    
    # YouTube Video Downloader
    st.subheader("📺 YouTube Video Downloader")
    youtube_url = st.text_input("Enter YouTube URL")
    if youtube_url:
        try:
            yt = YouTube(youtube_url)
            st.write(f"Title: {yt.title}")
            st.write(f"Views: {yt.views}")
            st.write(f"Duration: {yt.length} seconds")
            st.image(yt.thumbnail_url, use_column_width=True)
            
            if st.button("Download Video"):
                video = yt.streams.get_highest_resolution()
                video.download()
                st.success("Video downloaded successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Display generated files
    st.subheader("📁 Generated Files")
    files = list_files()
    if files:
        for file in files:
            with st.expander(file):
                display_file(f"generated_files/{file}")
    else:
        st.info("No files generated yet.")
    
    # Bottom bar with action buttons
    st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Reset All", key="reset_all"):
            st.session_state.clear()
            st.experimental_rerun()
    with col2:
        if st.button("🧹 Clear Code", key="clear_code"):
            st.session_state.code_editor = ""
            st.experimental_rerun()
    with col3:
        if st.button("💾 Save Code", key="save_code"):
            save_file(f"code_{int(time.time())}.py", st.session_state.code_editor)
            st.success("Code saved successfully!")
    with col4:
        if st.button("📊 Code Stats", key="code_stats"):
            code_lines = st.session_state.code_editor.split("\n")
            st.info(f"Lines of code: {len(code_lines)}")
            st.info(f"Characters: {len(st.session_state.code_editor)}")
    st.markdown('</div>', unsafe_allow_html=True)

# Run the enhanced app
if __name__ == "__main__":
    main()
