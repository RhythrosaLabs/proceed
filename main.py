import streamlit as st
import io
import contextlib
import openai

# OpenAI API Key (set your key here or use environment variables)
OPENAI_API_KEY = "your_openai_api_key_here"

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

# Function to call GPT-4o-mini (or any OpenAI GPT model)
def chat_with_gpt(prompt):
    """Send a prompt to GPT-4o-mini and return the response."""
    try:
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        message = response['choices'][0]['message']['content']
        return message
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
        prompt = st.text_area("Enter your prompt for GPT-4o-mini:", height=150)
        
        if st.button("Chat with GPT-4o-mini"):
            if prompt:
                response = chat_with_gpt(prompt)
                st.subheader("Response:")
                st.write(response)
            else:
                st.warning("Please enter a prompt.")

# Entry point
if __name__ == "__main__":
    main()
