import streamlit as st
import io
import contextlib
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

# Main function to run the Streamlit app
def main():
    st.title("Enhanced Python Code Executor")

    st.write(
        """
        This is an enhanced code execution environment. You can write Python code and it will be executed.
        - Imports for `matplotlib`, `pandas`, and `numpy` are automatically available.
        - You can generate plots and display DataFrames directly.
        """
    )
    
    # Code input area
    default_code = """# Example: Generate and plot data using matplotlib\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.title('Sine Wave')\nplt.xlabel('x')\nplt.ylabel('sin(x)')\nst.pyplot(plt)"""
    user_code = st.text_area("Python Code", value=default_code, height=300)

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
