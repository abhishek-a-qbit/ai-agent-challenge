import streamlit as st
import pandas as pd
import os
import io
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import importlib.util

# --- Agent and Parser Logic ---
# This section contains the core logic from the provided agent.py file.
# It has been integrated into the Streamlit app for a single-file demonstration.

# Ensure required libraries are available
try:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    import pdfplumber
except ImportError:
    st.error("""
    Please install the required libraries to run this agent:
    `pip install streamlit pandas langchain-openai langgraph pdfplumber`
    """)
    st.stop()

@dataclass
class AgentState:
    """State for the agent workflow"""
    target_bank: str
    pdf_path: str
    generated_code: str
    attempts: int
    errors: List[str]
    final_parser_path: str
    pdf_analysis: Dict[str, Any]

class PDFAnalyzer:
    """Analyzes PDF structure and extracts key information"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_api_key
        )
    
    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF structure and return key insights"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    tables = page.extract_tables()
                    pages.append({
                        'text': text,
                        'tables': tables,
                        'width': page.width,
                        'height': page.height
                    })
            
                # Analyze with LLM
                prompt = f"""
                Analyze this PDF structure and provide:
                1. Document type (bank statement, etc.)
                2. Key sections identified
                3. Table structures found
                4. Data patterns
                5. Recommended parsing approach
                
                PDF Info: {len(pages)} pages, tables: {[len(p.get('tables', [])) for p in pages]}
                
                First table structure: {tables[0][:3] if tables and tables[0] else 'No tables'}
                """
            
                response = self.llm.invoke(prompt)
                return {
                    'pages': pages,
                    'analysis': response.content,
                    'page_count': len(pages)
                }
        except Exception as e:
            return {'error': str(e)}

class CodeGenerator:
    """Generates Python parser code based on analysis"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_api_key
        )
    
    def generate_parser(self, pdf_analysis: Dict[str, Any], target_bank: str) -> str:
        """Generate parser code based on PDF analysis"""
        
        st.subheader("Debugging LLM Call")
        with st.expander("Show LLM Connectivity Test"):
            st.info("Testing LLM connectivity by generating random content...")
            test_prompt = "Generate a short, random paragraph about the future of technology."
            st.code(test_prompt)
            
            test_response = ""
            try:
                test_response = self.llm.invoke(test_prompt)
                if test_response and test_response.content:
                    st.success("Test Successful! LLM returned a response.")
                    st.write(test_response.content)
                else:
                    st.error("Test Failed! LLM returned an empty response.")
            except Exception as e:
                st.error(f"Test Failed! LLM invocation raised an exception: {e}")

        st.info("Input PDF Analysis:")
        st.json(pdf_analysis)
        
        prompt = f"""
        Generate a Python parser for {target_bank} bank statements.
        
        Requirements:
        1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
        2. Must return a pandas DataFrame containing the extracted transaction data.
        3. Use pdfplumber for PDF processing, focusing on table extraction.
        4. Handle errors gracefully.
        5. Include proper type hints and docstring.
        6. The PDF contains transaction tables that need to be extracted and formatted.
        
        PDF Analysis: {pdf_analysis.get('analysis', 'No analysis available')}
        
        Focus on:
        - Extracting tables from PDF pages.
        - Processing transaction data (Date, Description, Debit Amt, Credit Amt, Balance, etc.).
        - Handling empty cells and data formatting.
        - Return the resulting DataFrame.
        
        Generate complete, runnable code with imports.
        """
        
        st.info("Prompt sent to LLM for Parser Generation:")
        st.code(prompt)
        
        response = ""
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            st.error(f"LLM invocation failed: {e}")
            return ""

        st.info(f"LLM Response object: {repr(response)}")
        
        if not response or not response.content:
            st.warning("LLM response content is empty.")
            return ""

        st.info("Raw LLM output:")
        st.code(response.content)
        
        # Sanitize the output to get only the code block
        code = response.content
        match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if match:
            code = match.group(1)
            st.success("Successfully extracted code from code block.")
        else:
            # Fallback to the entire response if the code block is not found
            code = response.content
            st.warning("Could not find a Python code block. Using the full response content.")
        
        st.info(f"Content of 'generated_code' before being returned (length: {len(code)}):")
        st.code(code)
        
        return code

# Modified to accept openai_api_key
def analyze_pdf(state: AgentState, openai_api_key: str) -> AgentState:
    """Node: Analyze PDF structure"""
    st.info("🔍 Analyzing PDF structure...")
    analyzer = PDFAnalyzer(openai_api_key)
    analysis = analyzer.analyze_pdf_structure(state.pdf_path)
    
    if 'error' in analysis:
        state.errors.append(f"PDF analysis failed: {analysis['error']}")
        return state
    
    state.pdf_analysis = analysis
    st.success("✅ PDF analysis completed")
    return state

# Modified to accept openai_api_key
def generate_code(state: AgentState, openai_api_key: str) -> AgentState:
    """Node: Generate parser code"""
    st.info(f"💻 Generating parser code for {state.target_bank}...")
    
    if not hasattr(state, 'pdf_analysis'):
        state.errors.append("No PDF analysis available")
        return state
    
    generator = CodeGenerator(openai_api_key)
    code = generator.generate_parser(
        state.pdf_analysis, 
        state.target_bank
    )
    
    if not code.strip():
        state.errors.append("Code generation failed: The LLM returned an empty response.")
        return state

    # Correctly update the state with the generated code
    state.generated_code = code
    st.success("✅ Code generation completed")
    return state

# Modified the workflow to pass the API key to the nodes
def create_workflow(openai_api_key: str) -> StateGraph:
    """Create the agent workflow graph"""
    workflow = StateGraph(AgentState)
    
    # Pass the API key to the nodes
    workflow.add_node("analyze_pdf", lambda state: analyze_pdf(state, openai_api_key))
    workflow.add_node("generate_code", lambda state: generate_code(state, openai_api_key))
    workflow.add_node("save_parser", save_parser)
    
    workflow.add_edge(START, "analyze_pdf")
    workflow.add_edge("analyze_pdf", "generate_code")
    workflow.add_edge("generate_code", "save_parser")
    workflow.add_edge("save_parser", END)
    
    return workflow

def save_parser(state: AgentState) -> AgentState:
    """Node: Save the working parser"""
    st.info("💾 Saving parser...")
    
    parser_dir = Path("custom_parsers")
    parser_dir.mkdir(exist_ok=True)
    
    parser_path = parser_dir / f"{state.target_bank}_parser.py"
    with open(parser_path, 'w') as f:
        f.write(state.generated_code)
    
    state.final_parser_path = str(parser_path)
    st.success(f"✅ Parser saved to: {parser_path}")
    return state


def main():
    st.set_page_config(layout="wide", page_title="PDF Parser Generator")
    st.title("Agent-as-Coder: PDF Parser Generator")
    st.markdown("""
    This app demonstrates an autonomous agent that generates a Python parser for bank statement PDFs.
    
    **Instructions:**
    1.  **Paste your OpenAI API key below.**
    2.  Upload a sample PDF statement.
    3.  Provide a name for the target bank.
    4.  Click "Run Agent" to start the process.
    """)
    
    # --- New API Key Input Slot ---
    st.subheader("Enter OpenAI API Key")
    openai_api_key = st.text_input(
        label="OpenAI API Key", 
        type="password", 
        help="You can get a key from https://platform.openai.com/account/api-keys"
    )
    st.markdown("---")

    # --- File Uploads ---
    pdf_file = st.file_uploader("Upload PDF Statement", type="pdf")
    target_bank = st.text_input("Enter Target Bank Name (e.g., icici)", "icici")

    # --- Run Button ---
    if st.button("Run Agent"):
        if not openai_api_key:
            st.error("Error: The OpenAI API key is missing. Please enter your key.")
            return

        if not pdf_file or not target_bank:
            st.error("Please upload a PDF and provide a target bank name.")
            return
        
        st.header("Agent Execution Log")
        
        # Save uploaded file to a temporary location
        data_dir = "./temp_data"
        os.makedirs(data_dir, exist_ok=True)
        pdf_path = os.path.join(data_dir, pdf_file.name)
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Initialize the agent state
        initial_state = AgentState(
            target_bank=target_bank,
            pdf_path=pdf_path,
            generated_code="",
            attempts=0,
            errors=[],
            final_parser_path="",
            pdf_analysis={}
        )
        
        # Create and run workflow with the API key
        workflow = create_workflow(openai_api_key)
        app = workflow.compile()
        
        st.write(f"🚀 Starting agent for {target_bank} bank...")
        st.write(f"📄 PDF: {pdf_file.name}")
        st.write("-" * 50)
        
        try:
            final_state = app.invoke(initial_state)
            st.subheader("Final State after Workflow Completion:")
            st.json(final_state)
            
            if final_state.final_parser_path:
                st.balloons()
                st.success(f"\n🎉 Success! Parser generated at: {final_state.final_parser_path}")
                st.write("Final Code:")
                st.code(final_state.generated_code, language="python")
                
                # --- New Code for Parsing and Download ---
                st.header("Parsed Output")
                
                # Dynamically import the generated parser
                parser_path = final_state.final_parser_path
                spec = importlib.util.spec_from_file_location("dynamic_parser", parser_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for the 'parse' function before calling it
                if hasattr(module, 'parse'):
                    parsed_df = module.parse(final_state.pdf_path)
                    
                    st.subheader("Extracted Data")
                    st.dataframe(parsed_df)
                    
                    # Create a download button for the CSV
                    csv_buffer = io.StringIO()
                    parsed_df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue().encode('utf-8')
                    
                    st.download_button(
                        label="Download Parsed CSV",
                        data=csv_bytes,
                        file_name=f"{target_bank}_statement_parsed.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Error: The generated parser code does not contain a 'parse' function.")
                    st.warning("Please check the 'Final Code' section above for the generated code.")
                
            else:
                st.error(f"\n❌ Failed to generate parser.")
                st.write("Errors:")
                for error in final_state.errors:
                    st.error(error)
            
        except Exception as e:
            st.error(f"❌ Workflow failed: {e}")
            st.write("Please check the logs above for more details.")

if __name__ == "__main__":
    main()
