import streamlit as st
import io
import contextlib

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

# Main function to run the Streamlit app
def main():
    st.title("Code Executor App")
    
    # Code input area
    st.subheader("Write your Python code:")
    default_code = """# Write your Python code here\nprint('Hello, World!')"""
    user_code = st.text_area("Python Code", value=default_code, height=200)
    
    # Run code button
    if st.button("Run Code"):
        # Execute and capture the output
        output, error = execute_code(user_code)
        
        # Display the output or error
        if error:
            st.error(f"Error:\n{error}")
        else:
            st.subheader("Output:")
            st.code(output, language="python")

# Entry point
if __name__ == "__main__":
    main()
