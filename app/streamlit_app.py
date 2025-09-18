import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
from pathlib import Path

# Add the parent directory to the system path to allow importing agent.py
# This is required for the Streamlit app to find the agent's modules.
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the necessary components from the provided agent.py
# This assumes agent.py is in the same directory.
from agent import AgentState, analyze_pdf, generate_code, test_code, save_parser, should_retry, create_workflow

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Parser Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #004d40;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #1a76d2;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1a76d2;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e8b57;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #3cb371;
    }
</style>
""", unsafe_allow_html=True)

def run_agent_workflow(target_bank, pdf_path, csv_path):
    """
    Runs the agent workflow and provides updates to Streamlit.
    This function mimics the LangGraph flow in a sequential manner for the UI.
    """
    st.session_state.status_messages = []
    
    # Check if a Groq API key is set
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY environment variable is not set. Please set it and try again.")
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
        csv_schema = {
            'columns': list(df.columns),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'shape': df.shape
        }
    except Exception as e:
        st.error(f"❌ Failed to read CSV to determine schema: {e}")
        return None, None, None

    # Initialize agent state
    state = AgentState(
        target_bank=target_bank,
        pdf_path=pdf_path,
        csv_path=csv_path,
        csv_schema=csv_schema,
        generated_code="",
        test_results={},
        attempts=0,
        errors=[],
        final_parser_path=""
    )

    # Mimic the workflow steps
    st.markdown("<h3 class='section-header'>Step 1: Analyzing PDF</h3>", unsafe_allow_html=True)
    with st.spinner("Analyzing PDF structure..."):
        state = analyze_pdf(state)
        if state.errors:
            st.error(f"❌ Error during PDF analysis: {state.errors[-1]}")
            return None, None, state.errors
        st.success("✅ PDF analysis complete!")
        with st.expander("Show PDF Analysis Details"):
            st.json(state.pdf_analysis.get('analysis', 'No analysis data.'))

    st.markdown("<h3 class='section-header'>Step 2: Generating Parser Code</h3>", unsafe_allow_html=True)
    with st.spinner("Generating Python code for the parser..."):
        state = generate_code(state)
        if state.errors:
            st.error(f"❌ Error during code generation: {state.errors[-1]}")
            return None, None, state.errors
        st.success("✅ Code generation complete!")
        with st.expander("Show Generated Code"):
            st.code(state.generated_code, language='python')

    st.markdown("<h3 class='section-header'>Step 3: Testing the Code</h3>", unsafe_allow_html=True)
    with st.spinner("Running tests against the sample CSV..."):
        state = test_code(state)
        if state.test_results.get('is_equal'):
            st.success("✅ Code test passed! The generated parser works correctly.")
            st.markdown(f"""
                <div style='background-color: #d4edda; border-radius: 8px; padding: 15px; border-left: 5px solid #28a745;'>
                    <p style='margin: 0;'>🎉 **Success!** The generated parser correctly matched the expected CSV output.</p>
                    <p style='margin: 0;'>Rows: {state.test_results['result_shape'][0]} | Columns: {state.test_results['result_shape'][1]}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 class='section-header'>Step 4: Saving the Parser</h3>", unsafe_allow_html=True)
            with st.spinner("Saving the final parser file..."):
                state = save_parser(state)
                st.success(f"✅ Parser saved successfully to: {state.final_parser_path}")
            
            return state.generated_code, state.final_parser_path, None

        else:
            st.warning("⚠️ Code test failed. Attempting a self-correction loop...")
            
            # Simplified retry loop for the UI
            with st.spinner("Agent is self-correcting the code..."):
                state.attempts += 1
                st.info(f"🔄 Retry attempt {state.attempts}")
                state.errors.append(f"Test failed: {state.test_results.get('error', 'DataFrame mismatch')}")
                state = generate_code(state)
                state = test_code(state)
                
                if state.test_results.get('is_equal'):
                    st.success("✅ Self-correction successful! The corrected code now passes the tests.")
                    with st.expander("Show Corrected Code"):
                        st.code(state.generated_code, language='python')
                    
                    st.markdown("<h3 class='section-header'>Step 4: Saving the Corrected Parser</h3>", unsafe_allow_html=True)
                    with st.spinner("Saving the final corrected parser file..."):
                        state = save_parser(state)
                        st.success(f"✅ Corrected parser saved successfully to: {state.final_parser_path}")
                    
                    return state.generated_code, state.final_parser_path, None
                else:
                    st.error("❌ Final attempt failed. The agent could not generate a working parser.")
                    st.error(f"Final error: {state.test_results.get('error', 'DataFrame mismatch')}")
                    return None, None, state.errors
                    
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        return None, None, [str(e)]

def main():
    """Main Streamlit application entry point."""
    st.markdown('<h1 class="main-header">📄 PDF Parser Agent</h1>', unsafe_allow_html=True)
    st.markdown("### An interactive demonstration of a self-correcting AI agent for bank statement parsing.")
    
    st.info("💡 To use the agent, please upload a sample bank statement PDF and its corresponding CSV output.")
    
    with st.form("agent_form"):
        target_bank = st.text_input("Target Bank Name (e.g., 'icici', 'sbi')", "icici")
        uploaded_pdf = st.file_uploader("Upload Sample PDF", type="pdf")
        uploaded_csv = st.file_uploader("Upload Expected CSV", type="csv")
        
        submit_button = st.form_submit_button("Run Agent")

    if submit_button:
        if not all([uploaded_pdf, uploaded_csv]):
            st.warning("Please upload both a PDF and a CSV file.")
        else:
            # Create a temporary directory and save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                csv_path = os.path.join(temp_dir, uploaded_csv.name)
                with open(csv_path, "wb") as f:
                    f.write(uploaded_csv.getbuffer())

                st.markdown("---")
                generated_code, parser_path, errors = run_agent_workflow(target_bank, pdf_path, csv_path)
                st.markdown("---")
                
                if errors:
                    st.markdown('<h2 class="section-header">Agent Run Summary</h2>', unsafe_allow_html=True)
                    st.error("❌ Agent failed to generate a working parser.")
                    with st.expander("Show Errors"):
                        st.write(errors)
                elif parser_path:
                    st.markdown('<h2 class="section-header">Agent Run Summary</h2>', unsafe_allow_html=True)
                    st.success(f"✅ Agent successfully generated a working parser at `{parser_path}`.")
                    st.markdown("You can find the generated code below:")
                    st.code(generated_code, language='python')
                    
if __name__ == "__main__":
    main()
