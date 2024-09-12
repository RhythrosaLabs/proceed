import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import pydeck as pdk
import networkx as nx
import scipy.stats as stats
from sklearn import datasets, model_selection, preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import tensorflow as tf
from PIL import Image
import cv2
import io
import base64
import requests
import json
import sys
import traceback
import ast
import astor
import black
import time
import threading
import queue
import sqlite3
from streamlit_ace import st_ace
from streamlit_lottie import st_lottie
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.models.widgets import Button
from bokeh.models import CustomJS

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Super Enhanced Code Executor", page_icon="🚀")

# Custom CSS for an even sleeker look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
    }
    .Widget>label {
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #ffffff;
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        background-color: #262730;
        color: #ffffff;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #ffffff;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .output-area {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .code-editor {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        overflow: hidden;
    }
    .execution-info {
        background-color: #2e7bcf;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .st-emotion-cache-1rtdyuf {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'favorite_snippets' not in st.session_state:
    st.session_state.favorite_snippets = []

# Sidebar for settings and options
with st.sidebar:
    st.title("⚙️ Settings & Tools")
    
    # Animated logo
    st_lottie("https://assets5.lottiefiles.com/packages/lf20_sz668bpv.json", height=200)
    
    execution_mode = st.selectbox(
        "Execution Mode",
        ["Python Script", "Notebook Style", "Data Analysis", "Machine Learning", "Deep Learning", "Computer Vision", "Web Scraping", "API Integration"]
    )
    
    if execution_mode in ["Data Analysis", "Machine Learning", "Deep Learning"]:
        dataset = st.selectbox(
            "Choose a dataset",
            ["Iris", "Boston Housing", "Breast Cancer", "Wine", "Digits", "Custom CSV"]
        )
        if dataset == "Custom CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.session_state.custom_data = data
    
    # Code optimization options
    st.subheader("Code Optimization")
    optimize_imports = st.checkbox("Optimize Imports", value=True)
    format_code = st.checkbox("Format Code (Black)", value=True)
    
    # Performance profiling
    st.subheader("Performance Profiling")
    enable_profiling = st.checkbox("Enable Profiling", value=False)
    
    # Collaboration features
    st.subheader("Collaboration")
    enable_collab = st.checkbox("Enable Real-time Collaboration", value=False)
    if enable_collab:
        collab_room = st.text_input("Collaboration Room ID")

# Main area
st.title("🚀 Super Enhanced Code Executor")

# Code input area with syntax highlighting
st.subheader("📝 Code Input")
code = st_ace(
    placeholder="Enter your Python code here",
    language="python",
    theme="monokai",
    keybinding="vscode",
    font_size=14,
    tab_size=4,
    show_gutter=True,
    show_print_margin=True,
    wrap=False,
    auto_update=True,
    readonly=False,
    min_lines=20,
    key="code_editor"
)

# Execute button with loading animation
if st.button("🔥 Execute Code", key="execute_button"):
    with st.spinner("Executing code..."):
        start_time = time.time()
        
        # Create a capture object to redirect stdout and stderr
        captured_output = io.StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output
        
        try:
            # Optimize imports if enabled
            if optimize_imports:
                tree = ast.parse(code)
                optimized_tree = ast.fix_missing_locations(ast.parse(astor.to_source(tree)))
                code = astor.to_source(optimized_tree)
            
            # Format code if enabled
            if format_code:
                code = black.format_str(code, mode=black.FileMode())
            
            # Execute the code based on the selected mode
            if execution_mode == "Python Script":
                exec(code)
            elif execution_mode == "Notebook Style":
                exec(code, globals())
            elif execution_mode in ["Data Analysis", "Machine Learning", "Deep Learning"]:
                # Load the selected dataset
                if dataset == "Custom CSV":
                    data = st.session_state.custom_data
                else:
                    data = getattr(datasets, f"load_{dataset.lower().replace(' ', '_')}")()
                
                X = pd.DataFrame(data.data, columns=data.feature_names)
                y = pd.Series(data.target, name="target")
                
                # Make the data available in the execution environment
                exec(f"X = {X.to_dict()}")
                exec(f"y = {y.to_dict()}")
                
                # Execute the user's code
                exec(code, globals())
            elif execution_mode == "Computer Vision":
                # Load a sample image for computer vision tasks
                sample_image = Image.open("sample_image.jpg")
                exec("sample_image = sample_image", globals())
                exec(code, globals())
            elif execution_mode == "Web Scraping":
                exec(f"import requests\nfrom bs4 import BeautifulSoup\n{code}", globals())
            elif execution_mode == "API Integration":
                exec(f"import requests\nimport json\n{code}", globals())
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.code(traceback.format_exc(), language="python")
        
        # Reset stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display the captured output
        st.subheader("🖥️ Output")
        with st.expander("View Raw Output", expanded=True):
            st.code(captured_output.getvalue(), language="python")
        
        # Display execution information
        st.markdown(f"""
        <div class="execution-info">
            <strong>Execution Time:</strong> {execution_time:.2f} seconds<br>
            <strong>Lines of Code:</strong> {len(code.split('\n'))}
        </div>
        """, unsafe_allow_html=True)
        
        # Add to execution history
        st.session_state.execution_history.append({
            "code": code,
            "output": captured_output.getvalue(),
            "execution_time": execution_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

# Display any generated plots
if "plt" in globals():
    st.pyplot(plt)

if "px" in globals():
    st.plotly_chart(px)

if "alt" in globals():
    st.altair_chart(alt)

if "pdk" in globals():
    st.pydeck_chart(pdk)

# Display any generated DataFrames
for var in globals():
    if isinstance(globals()[var], pd.DataFrame):
        st.subheader(f"DataFrame: {var}")
        st.dataframe(globals()[var])

# Add a feature to save the executed code
if st.button("💾 Save Code"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"code_execution_{timestamp}.py"
    with open(filename, "w") as f:
        f.write(code)
    st.success(f"Code saved as {filename}")

# Add a feature to load code from a file
uploaded_file = st.file_uploader("📂 Load code from file", type=["py", "txt"])
if uploaded_file is not None:
    content = uploaded_file.read().decode()
    st.text_area("Loaded Code:", value=content, height=300, key="loaded_code")
    if st.button("Use Loaded Code"):
        st.session_state.code_editor = content

# Execution History
with st.expander("📜 Execution History"):
    for idx, execution in enumerate(reversed(st.session_state.execution_history)):
        st.markdown(f"**Execution {len(st.session_state.execution_history) - idx}**")
        st.markdown(f"Timestamp: {execution['timestamp']}")
        st.markdown(f"Execution Time: {execution['execution_time']:.2f} seconds")
        if st.button(f"View Code {len(st.session_state.execution_history) - idx}"):
            st.code(execution['code'], language="python")
        if st.button(f"View Output {len(st.session_state.execution_history) - idx}"):
            st.code(execution['output'], language="python")
        st.markdown("---")

# Favorite Code Snippets
with st.expander("⭐ Favorite Snippets"):
    new_snippet = st.text_area("Add a new favorite snippet:")
    if st.button("Add Snippet"):
        st.session_state.favorite_snippets.append(new_snippet)
    
    for idx, snippet in enumerate(st.session_state.favorite_snippets):
        st.code(snippet, language="python")
        if st.button(f"Use Snippet {idx + 1}"):
            st.session_state.code_editor = snippet

# Add a help section
with st.expander("ℹ️ Help & Documentation"):
    st.markdown("""
    ## How to use the Super Enhanced Code Executor

    1. Choose an execution mode from the sidebar.
    2. If applicable, select a dataset for data analysis or machine learning tasks.
    3. Enter your Python code in the code editor.
    4. Use the optimization options in the sidebar to improve your code.
    5. Click the "Execute Code" button to run your code.
    6. View the output, including any printed results, generated plots, or DataFrames.
    7. Save your code, load existing code, or use favorite snippets.
    8. Check the execution history to review past runs.

    ### Available Libraries
    - Data manipulation: pandas, numpy
    - Visualization: matplotlib, seaborn, plotly, altair, bokeh
    - Machine Learning: scikit-learn, tensorflow
    - Computer Vision: PIL, OpenCV
    - Web Scraping: requests, BeautifulSoup
    - API Integration: requests, json

    ### Keyboard Shortcuts
    - Ctrl + Enter: Execute code
    - Ctrl + S: Save code
    - Ctrl + O: Open file

    For more information, please refer to the documentation of each library.
    """)

# Real-time collaboration (placeholder)
if enable_collab:
    st.sidebar.subheader("🤝 Collaboration")
    st.sidebar.markdown(f"Room: {collab_room}")
    st.sidebar.markdown("Connected users: 1")  # Placeholder

# Performance profiling results (placeholder)
if enable_profiling:
    st.sidebar.subheader("🚀 Performance Profile")
    st.sidebar.markdown("Top time-consuming functions:")
    st.sidebar.markdown("1. function_a: 0.5s")
    st.sidebar.markdown("2. function_b: 0.3s")
    st.sidebar.markdown("3. function_c: 0.1s")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Your Name | [GitHub](https://github.com/yourusername) | [Documentation](https://yourdocs.com)")

# Add a chat-like interface for AI assistance
st.sidebar.subheader("🤖 AI Assistant")
user_question = st.sidebar.text_input("Ask a coding question:")
if st.sidebar.button("Get Help"):
    # Placeholder for AI response (you'd typically call an API here)
    ai_response = "Here's a suggestion for your code: ..."
    st.sidebar.markdown(f"AI: {ai_response}")

# Bokeh event for handling real-time updates
bokeh_button = Button(label="Trigger Update")
bokeh_button.js_on_event("button_click", CustomJS(code="""
    document.dispatchEvent(new CustomEvent("streamlit:trigger-update"));
"""))
streamlit_bokeh_events(bokeh_button, events="streamlit:trigger-update", key="update", refresh_on_update=True)
