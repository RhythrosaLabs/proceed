import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_ace import st_ace
import json

# Custom CSS to make the UI more appealing
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
    .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
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

# Main function to run the Streamlit app
def main():
    st.title("🚀 Code Executor & GPT-4 Chat")
    
    # Load Lottie animations
    lottie_code = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
    lottie_chat = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_lntyk83o.json")

    # Sidebar tabs with icons
    option = st.sidebar.selectbox("Select Mode", ["💻 Code Executor", "🤖 GPT-4 Chat"])

    if option == "💻 Code Executor":
        st.subheader("Write and Execute Code")
        st_lottie(lottie_code, height=200, key="code_animation")
        
        # Use Ace editor for code input
        user_code = st_ace(
            value="# Example: Generate a random dataset and plot it\n"
                  "import pandas as pd\n"
                  "import numpy as np\n"
                  "import plotly.express as px\n\n"
                  "# Generate random data\n"
                  "np.random.seed(10)\n"
                  "data = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})\n\n"
                  "# Create Plotly figure\n"
                  "fig = px.scatter(data, x='x', y='y', title='Random Scatter Plot')\n"
                  "fig.update_layout(template='plotly_dark')\n"
                  "fig",
            language="python",
            theme="monokai",
            height=300
        )

        if st.button("🚀 Run Code"):
            with st.spinner("Executing code..."):
                try:
                    result = execute_code(user_code)
                    st.subheader("Visualization Output")
                    if "fig" in result:
                        st.plotly_chart(result["fig"], use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.error("No figure found. Make sure to assign a Plotly figure to the variable `fig`.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif option == "🤖 GPT-4 Chat":
        st.subheader("Chat with GPT-4")
        st_lottie(lottie_chat, height=200, key="chat_animation")
        
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        prompt = st.text_area("Enter your prompt for GPT-4:", height=150)
        
        if st.button("💬 Chat with GPT-4"):
            if not api_key:
                st.warning("Please enter a valid OpenAI API key.")
            elif not prompt:
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Generating response..."):
                    response = chat_with_gpt(prompt, api_key)
                    st.subheader("GPT-4 Response:")
                    st.markdown(response)

# Entry point
if __name__ == "__main__":
    main()
