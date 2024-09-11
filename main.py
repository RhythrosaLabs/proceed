import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np

# Function to execute user-provided code
def execute_code(code):
    local_vars = {}
    exec(code, {}, local_vars)
    return local_vars

# Function to call GPT-4o-mini (or any OpenAI GPT model) via requests
def chat_with_gpt(prompt, api_key):
    """Send a prompt to GPT-4 via a requests call and return the response."""
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
        response.raise_for_status()  # Raise error for bad status codes
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Main function to run the Streamlit app
def main():
    st.title("Code Executor & GPT-4o-mini Chat with Tabs")

    # Create tabs for API and Chat
    tabs = st.tabs(["API", "Chat"])

    # Tab 1: API - Code Executor
    with tabs[0]:
        st.subheader("Write and Execute Code")
        st.write("Write Python code and run it. You can visualize data using Plotly, Altair, and more.")

        # Default example code for visualization
        default_code = """# Example: Generate a random dataset and plot it\nimport pandas as pd\nimport numpy as np\nimport plotly.express as px\n\n# Generate random data\nnp.random.seed(10)\ndata = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})\n\n# Create Plotly figure\nfig = px.scatter(data, x='x', y='y', title='Random Scatter Plot')\nfig"""
        user_code = st.text_area("Python Code", default_code, height=300)

        # Button to run the code
        if st.button("Run Code", key="run_code"):
            try:
                result = execute_code(user_code)
                st.subheader("Visualization Output")
                if "fig" in result:
                    st.plotly_chart(result["fig"], use_container_width=True)
                else:
                    st.error("No figure found. Make sure to assign a Plotly figure to the variable `fig`.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Tab 2: Chat with GPT-4o-mini
    with tabs[1]:
        st.subheader("Chat with GPT-4o-mini")
        st.write("Enter a prompt below to chat with GPT-4o-mini.")

        # API Key Input
        api_key = st.text_input("Enter your OpenAI API Key", type="password")

        # Prompt Input
        prompt = st.text_area("Enter your prompt for GPT-4o-mini:", height=150)

        if st.button("Chat with GPT-4o-mini", key="chat_gpt"):
            if not api_key:
                st.warning("Please enter a valid OpenAI API key.")
            elif not prompt:
                st.warning("Please enter a prompt.")
            else:
                response = chat_with_gpt(prompt, api_key)
                st.subheader("GPT-4o-mini Response:")
                st.write(response)

# Entry point
if __name__ == "__main__":
    main()
