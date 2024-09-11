import streamlit as st
import io
import contextlib
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a dictionary to store the execution context
execution_context = {}

# Function to execute user-provided code
def execute_code(code):
    """Executes user-provided Python code and captures output."""
    global execution_context
    output_buffer = io.StringIO()  # Capture standard output
    error_message = None

    try:
        with contextlib.redirect_stdout(output_buffer):
            # Execute the code in the context of the global execution dictionary
            exec(code, execution_context)
    except Exception as e:
        error_message = str(e)

    return output_buffer.getvalue(), error_message

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
    st.title("Enhanced Code Executor & GPT-4o-mini Chat App")

    st.write(
        """
        This is a combined environment for writing Python code and chatting with GPT-4o-mini.
        - Imports for `matplotlib`, `pandas`, and `numpy` are automatically available.
        - You can generate plots and display DataFrames directly.
        """
    )

    # Tabs for Code Executor and GPT-4o-mini Chat
    tabs = st.tabs(["Code Executor", "GPT-4o-mini Chat"])

    # Code Executor Tab
    with tabs[0]:
        st.subheader("Write and Execute Python Code")
        default_code = """# Example: Generate and plot data using matplotlib\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.title('Sine Wave')\nplt.xlabel('x')\nplt.ylabel('sin(x)')\nst.pyplot(plt)"""
        user_code = st.text_area("Python Code", value=default_code, height=300)

        if st.button("Run Code"):
            # Execute and capture the output
            output, error = execute_code(user_code)

            # Display the output or error
            if error:
                st.error(f"Error:\n{error}")
            else:
                st.subheader("Output:")
                st.code(output, language="python")

    # GPT-4o-mini Chat Tab
    with tabs[1]:
        st.subheader("Chat with GPT-4o-mini")

        # API Key Input
        api_key = st.text_input("Enter your OpenAI API Key", type="password")

        # Prompt Input
        prompt = st.text_area("Enter your prompt for GPT-4o-mini:", height=150)

        if st.button("Chat with GPT-4o-mini"):
            if not api_key:
                st.warning("Please enter a valid OpenAI API key.")
            elif not prompt:
                st.warning("Please enter a prompt.")
            else:
                response = chat_with_gpt(prompt, api_key)
                st.subheader("Response:")
                st.write(response)

# Entry point
if __name__ == "__main__":
    main()
