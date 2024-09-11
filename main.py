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
    .chat-message.bot {
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
        st.markdown(content)

# Main function to run the Streamlit app
def main():
    st.title("🚀 Conversational Coding Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your coding assistant. How can I help you today?"})

    # Sidebar for API key input
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Enter your OpenAI API key")
        st.markdown("2. Chat with the AI about coding tasks")
        st.markdown("3. Use '/code' to switch to code mode")
        st.markdown("4. Use '/run' to execute the code")
        st.markdown("5. Use '/clear' to start a new conversation")

    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Chat input
    prompt = st.chat_input("What's on your mind?")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        if prompt.lower() == "/clear":
            st.session_state.messages = []
            st.experimental_rerun()

        elif prompt.lower() == "/code":
            # Switch to code mode
            code = st_ace(
                value="# Enter your Python code here\n",
                language="python",
                theme="monokai",
                height=300
            )
            st.session_state.messages.append({"role": "assistant", "content": "Here's a code editor for you. Type your code and use '/run' to execute it."})
            display_chat_message("assistant", "Here's a code editor for you. Type your code and use '/run' to execute it.")

        elif prompt.lower() == "/run":
            # Execute the last code block
            last_code_block = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user" and msg["content"].startswith("```python") and msg["content"].endswith("```")), None)
            if last_code_block:
                code_to_run = last_code_block.strip("```python").strip()
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
                st.session_state.messages.append({"role": "assistant", "content": "No code block found to execute. Please enter a code block first."})
                display_chat_message("assistant", "No code block found to execute. Please enter a code block first.")

        else:
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
