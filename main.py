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

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Enhanced Code Executor", page_icon="🚀")

# Custom CSS for a sleeker look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #ffffff;
    }
    .stTextArea>div>div>textarea {
        background-color: #262730;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #ffffff;
    }
    .output-area {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 10px;
    }
    .code-editor {
        border: 1px solid #4CAF50;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings and options
with st.sidebar:
    st.title("⚙️ Settings")
    
    execution_mode = st.selectbox(
        "Execution Mode",
        ["Python Script", "Notebook Style", "Data Analysis", "Machine Learning", "Deep Learning", "Computer Vision"]
    )
    
    if execution_mode in ["Data Analysis", "Machine Learning", "Deep Learning"]:
        dataset = st.selectbox(
            "Choose a dataset",
            ["Iris", "Boston Housing", "Breast Cancer", "Wine", "Digits"]
        )

# Main area
st.title("🚀 Enhanced Code Executor")

# Code input area
st.subheader("📝 Code Input")
code = st.text_area("Enter your code here:", height=300, key="code_input")

# Execute button
if st.button("🔥 Execute Code"):
    st.subheader("🖥️ Output")
    
    # Create a capture object to redirect stdout and stderr
    captured_output = io.StringIO()
    sys.stdout = captured_output
    sys.stderr = captured_output
    
    try:
        # Execute the code based on the selected mode
        if execution_mode == "Python Script":
            exec(code)
        elif execution_mode == "Notebook Style":
            exec(code, globals())
        elif execution_mode in ["Data Analysis", "Machine Learning", "Deep Learning"]:
            # Load the selected dataset
            if dataset == "Iris":
                data = datasets.load_iris()
            elif dataset == "Boston Housing":
                data = datasets.load_boston()
            elif dataset == "Breast Cancer":
                data = datasets.load_breast_cancer()
            elif dataset == "Wine":
                data = datasets.load_wine()
            elif dataset == "Digits":
                data = datasets.load_digits()
            
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
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(traceback.format_exc())
    
    # Reset stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    # Display the captured output
    st.code(captured_output.getvalue(), language="python")

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        st.session_state.code_input = content

# Add a help section
with st.expander("ℹ️ Help"):
    st.markdown("""
    ## How to use the Enhanced Code Executor

    1. Choose an execution mode from the sidebar.
    2. If applicable, select a dataset for data analysis or machine learning tasks.
    3. Enter your Python code in the text area.
    4. Click the "Execute Code" button to run your code.
    5. View the output, including any printed results, generated plots, or DataFrames.
    6. Save your code or load existing code using the provided buttons.

    ### Available Libraries
    - Data manipulation: pandas, numpy
    - Visualization: matplotlib, seaborn, plotly, altair
    - Machine Learning: scikit-learn, tensorflow
    - Computer Vision: PIL, OpenCV

    For more information, please refer to the documentation of each library.
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Your Name | [GitHub](https://github.com/yourusername)")
