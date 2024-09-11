import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt

# Create a function to run code and display the output in real-time
def execute_code(code):
    local_vars = {}
    exec(code, {}, local_vars)
    return local_vars

def main():
    st.title("Code Execution with Visualization")
    
    # Columns for code input and output visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Write and Execute Code")
        
        # Default example code for visualization
        default_code = """# Example: Generate a random dataset and plot it\nimport pandas as pd\nimport numpy as np\nimport plotly.express as px\n\n# Generate random data\nnp.random.seed(10)\ndata = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})\n\n# Create Plotly figure\nfig = px.scatter(data, x='x', y='y', title='Random Scatter Plot')\nfig"""
        user_code = st.text_area("Python Code", default_code, height=300)

        # Button to run the code
        if st.button("Run Code"):
            try:
                result = execute_code(user_code)
                with col2:
                    st.subheader("Visualization Output")
                    if "fig" in result:
                        st.plotly_chart(result["fig"], use_container_width=True)
                    else:
                        st.error("No figure found. Make sure to assign a Plotly figure to the variable `fig`.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Entry point
if __name__ == "__main__":
    main()
