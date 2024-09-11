import streamlit as st
import io
import contextlib
import requests

# Function to execute user-provided code
def execute_code(code):
    """Executes user-provided Python code and captures output."""
    output_buffer = io.StringIO()
    error_message = None

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code)
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
    st.title("Code Executor & GPT-4o-mini Chat App")

    # Tab layout for Code Execution and GPT-4o-mini Chat
    tabs = st.tabs(["Code Executor", "GPT-4o-mini Chat"])

    with tabs[0]:
        st.subheader("Write and Execute Python Code:")
        default_code = """# Write your Python code here\nprint('Hello, World!')"""
        user_code = st.text_area("Python Code", value=default_code, height=200)

        if st.button("Run Code"):
            # Execute the code and display the result
            output, error = execute_code(user_code)
            if error:
                st.error(f"Error:\n{error}")
            else:
                st.subheader("Output:")
                st.code(output, language="python")

    with tabs[1]:
        st.subheader("Chat with GPT-4o-mini:")

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
