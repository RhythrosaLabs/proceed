import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_ace import st_ace
import json

# Custom CSS for the integrated UI
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
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
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
    .floating-button {
        position: fixed;
        right: 20px;
        bottom: 20px;
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
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to execute user-provided code
def execute_code(code):
    local_vars = {}
    exec(code, globals(), local_vars)
    return local_vars

# Function to call GPT-4 via requests
def chat_with_gpt(prompt, api_key):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
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

# Main function to run the Streamlit app
def main():
    st.title("🚀 Conversational Coding Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your coding assistant. How can I help you today? Feel free to ask questions, request code samples, or ask for explanations."})

    # Sidebar for API key input
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        st.markdown("---")
        st.markdown("### Quick Tips:")
        st.markdown("1. Chat naturally about coding tasks")
        st.markdown("2. Request code samples or explanations")
        st.markdown("3. Click 'Run Code' to execute any code block")
        st.markdown("4. Use 'Clear Chat' to start over")

    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Chat input
    prompt = st.chat_input("Ask me anything about coding...")

    # Floating action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏃‍♂️ Run Code", key="run_code"):
            last_code_block = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "assistant" and "```python" in msg["content"]), None)
            if last_code_block:
                code_to_run = last_code_block.split("```python")[1].split("```")[0].strip()
                with st.spinner("Executing code..."):
                    try:
                        result = execute_code(code_to_run)
                        output = "Code executed successfully."
                        if "fig" in result:
                            st.plotly_chart(result["fig"], use_container_width=True, config={'displayModeBar': False})
                            output += "\n\nVisualization displayed above."
                        st.session_state.messages.append({"role": "assistant", "content": output})
                        display_chat_message("assistant", output)
                    except Exception as e:
                        error_msg = f"Error executing code: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        display_chat_message("assistant", error_msg)
            else:
                st.warning("No code block found to execute. Please request a code sample first.")

    with col2:
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        # Process with GPT-4
        if api_key:
            with st.spinner("Thinking..."):
                response = chat_with_gpt(prompt, api_key)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
        else:
            st.warning("Please enter a valid OpenAI API key in the sidebar.")

# Entry point
if __name__ == "__main__":
    main()
