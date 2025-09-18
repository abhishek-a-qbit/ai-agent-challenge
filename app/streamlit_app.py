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
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    import pdfplumber
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    st.error("""
    Please install the required libraries to run this agent:
    `pip install streamlit pandas langchain-groq langgraph pdfplumber python-dotenv`
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

class PDFAnalyzer:
    """Analyzes PDF structure and extracts key information"""
    
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key
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
                
                Table Analysis:
                {[f'Page {i+1}: {len(p.get("tables", []))} tables, {len(p.get("tables", [])[0]) if p.get("tables") else 0} columns' for i, p in enumerate(pages)]}
                
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
    
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key
        )
    
    def generate_parser(self, pdf_analysis: Dict[str, Any], target_bank: str) -> str:
        """Generate parser code based on PDF analysis"""
        
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
        
        response = self.llm.invoke(prompt)
        return response.content

def analyze_pdf(state: AgentState) -> AgentState:
    """Node: Analyze PDF structure"""
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    st.info("🔍 Analyzing PDF structure...")
    analyzer = PDFAnalyzer(groq_api_key)
    analysis = analyzer.analyze_pdf_structure(state.pdf_path)
    
    if 'error' in analysis:
        state.errors.append(f"PDF analysis failed: {analysis['error']}")
        return state
    
    state.pdf_analysis = analysis
    st.success("✅ PDF analysis completed")
    return state

def generate_code(state: AgentState) -> AgentState:
    """Node: Generate parser code"""
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    st.info(f"💻 Generating parser code for {state.target_bank}...")
    
    if not hasattr(state, 'pdf_analysis'):
        state.errors.append("No PDF analysis available")
        return state
    
    generator = CodeGenerator(groq_api_key)
    code = generator.generate_parser(
        state.pdf_analysis, 
        state.target_bank
    )
    
    state.generated_code = code
    st.success("✅ Code generation completed")
    return state

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

def create_workflow() -> StateGraph:
    """Create the agent workflow graph"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_pdf", analyze_pdf)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("save_parser", save_parser)
    
    workflow.add_edge(START, "analyze_pdf")
    workflow.add_edge("analyze_pdf", "generate_code")
    workflow.add_edge("generate_code", "save_parser")
    workflow.add_edge("save_parser", END)
    
    return workflow

# --- Streamlit UI and Main Function ---

def main():
    st.title("Agent-as-Coder: PDF Parser Generator")
    st.markdown("""
    This app demonstrates an autonomous agent that generates a Python parser for bank statement PDFs.
    
    **Instructions:**
    1.  Upload a sample PDF statement.
    2.  Provide a name for the target bank.
    3.  Click "Run Agent" to start the process.
    
    **Note:** This agent requires a valid Groq API key to generate code, which can be set as a Streamlit secret.
    """)
    
    # --- File Uploads ---
    pdf_file = st.file_uploader("Upload PDF Statement", type="pdf")
    target_bank = st.text_input("Enter Target Bank Name (e.g., icici)", "icici")

    # --- Run Button ---
    if st.button("Run Agent"):
        if not pdf_file or not target_bank:
            st.error("Please upload a PDF and provide a target bank name.")
            return

        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Error: The GROQ_API_KEY is not set. Please set it as a Streamlit secret.")
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
            final_parser_path=""
        )
        
        # Create and run workflow
        workflow = create_workflow()
        app = workflow.compile()
        
        st.write(f"🚀 Starting agent for {target_bank} bank...")
        st.write(f"📄 PDF: {pdf_file.name}")
        st.write("-" * 50)
        
        try:
            final_state = app.invoke(initial_state)
            
            if 'final_parser_path' in final_state and final_state['final_parser_path']:
                st.balloons()
                st.success(f"\n🎉 Success! Parser generated at: {final_state['final_parser_path']}")
                st.write("Final Code:")
                st.code(final_state['generated_code'], language="python")
                
                # --- New Code for Parsing and Download ---
                st.header("Parsed Output")
                
                # Dynamically import the generated parser
                parser_path = final_state['final_parser_path']
                spec = importlib.util.spec_from_file_location("dynamic_parser", parser_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Parse the PDF using the generated code
                parsed_df = module.parse(final_state['pdf_path'])
                
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
                st.error(f"\n❌ Failed to generate parser.")
                st.write("Errors:")
                for error in final_state['errors']:
                    st.error(error)
        
        except Exception as e:
            st.error(f"❌ Workflow failed: {e}")

if __name__ == "__main__":
    main()
