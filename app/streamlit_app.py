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
    csv_path: str
    csv_schema: Dict[str, Any]
    generated_code: str
    test_results: Dict[str, Any]
    attempts: int
    errors: List[str]
    final_parser_path: str

class PDFAnalyzer:
    """Analyzes PDF structure and extracts key information"""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
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
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def generate_parser(self, pdf_analysis: Dict[str, Any], csv_schema: Dict[str, Any], target_bank: str) -> str:
        """Generate parser code based on PDF analysis and CSV schema"""
        
        prompt = f"""
        Generate a Python parser for {target_bank} bank statements.
        
        Requirements:
        1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
        2. Must return DataFrame matching this schema: {csv_schema}
        3. Use pdfplumber for PDF processing, focus on table extraction
        4. Handle errors gracefully
        5. Include proper type hints and docstring
        6. The PDF contains transaction tables that need to be extracted and formatted
        
        Expected CSV Schema: {csv_schema}
        PDF Analysis: {pdf_analysis.get('analysis', 'No analysis available')}
        
        Focus on:
        - Extracting tables from PDF pages
        - Processing transaction data (Date, Description, Debit Amt, Credit Amt, Balance)
        - Handling empty cells and data formatting
        - Ensuring output matches the expected CSV structure exactly
        
        Generate complete, runnable code with imports.
        """
        
        response = self.llm.invoke(prompt)
        return response.content

class CodeTester:
    """Tests generated code and provides feedback"""
    
    def test_parser(self, code: str, pdf_path: str, csv_path: str) -> Dict[str, Any]:
        """Test the generated parser code"""
        try:
            # Create temporary test file
            test_file = "temp_parser.py"
            with open(test_file, 'w') as f:
                f.write(code)
            
            # Import and test
            spec = importlib.util.spec_from_file_location("temp_parser", test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test parsing
            result_df = module.parse(pdf_path)
            expected_df = pd.read_csv(csv_path)
            
            # Compare DataFrames
            is_equal = result_df.equals(expected_df)
            
            # Cleanup
            os.remove(test_file)
            
            return {
                'success': True,
                'is_equal': is_equal,
                'result_shape': result_df.shape,
                'expected_shape': expected_df.shape,
                'result_columns': list(result_df.columns),
                'expected_columns': list(expected_df.columns)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'is_equal': False
            }

def analyze_pdf(state: AgentState) -> AgentState:
    """Node: Analyze PDF structure"""
    st.info("🔍 Analyzing PDF structure...")
    analyzer = PDFAnalyzer()
    analysis = analyzer.analyze_pdf_structure(state.pdf_path)
    
    if 'error' in analysis:
        state.errors.append(f"PDF analysis failed: {analysis['error']}")
        return state
    
    state.pdf_analysis = analysis
    st.success("✅ PDF analysis completed")
    return state

def generate_code(state: AgentState) -> AgentState:
    """Node: Generate parser code"""
    st.info(f"💻 Generating parser code for {state.target_bank}...")
    
    if not hasattr(state, 'pdf_analysis'):
        state.errors.append("No PDF analysis available")
        return state
    
    generator = CodeGenerator()
    code = generator.generate_parser(
        state.pdf_analysis, 
        state.csv_schema, 
        state.target_bank
    )
    
    state.generated_code = code
    st.success("✅ Code generation completed")
    return state

def test_code(state: AgentState) -> AgentState:
    """Node: Test generated code"""
    st.info("🧪 Testing generated code...")
    
    tester = CodeTester()
    test_results = tester.test_parser(
        state.generated_code,
        state.pdf_path,
        state.csv_path
    )
    
    state.test_results = test_results
    
    if test_results.get('success') and test_results.get('is_equal'):
        st.success("✅ Code test passed!")
        return state
    else:
        state.errors.append(f"Test failed: {test_results.get('error', 'DataFrame mismatch')}")
        st.error("❌ Code test failed")
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

def should_retry(state: AgentState) -> str:
    """Router: Decide whether to retry or end"""
    if state.attempts >= 3:
        return "end"
    
    if state.test_results.get('success') and state.test_results.get('is_equal'):
        return "save"
    
    return "retry"

def retry_with_feedback(state: AgentState) -> AgentState:
    """Node: Retry code generation with feedback"""
    state.attempts += 1
    st.warning(f"🔄 Retry attempt {state.attempts}/3")
    
    generator = CodeGenerator()
    
    feedback_prompt = f"""
    Previous code failed with error: {state.errors[-1]}
    Test results: {state.test_results}
    
    Generate improved code that fixes these issues.
    """
    
    improved_code = generator.llm.invoke(feedback_prompt).content
    state.generated_code = improved_code
    state.errors = []  # Clear previous errors
    
    return state

def create_workflow() -> StateGraph:
    """Create the agent workflow graph"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_pdf", analyze_pdf)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("test_code", test_code)
    workflow.add_node("save_parser", save_parser)
    workflow.add_node("retry", retry_with_feedback)
    
    workflow.add_edge(START, "analyze_pdf")
    workflow.add_edge("analyze_pdf", "generate_code")
    workflow.add_edge("generate_code", "test_code")
    workflow.add_conditional_edges(
        "test_code",
        should_retry,
        {
            "save": "save_parser",
            "retry": "retry",
            "end": END
        }
    )
    workflow.add_edge("retry", "test_code")
    workflow.add_edge("save_parser", END)
    
    return workflow

# --- Streamlit UI and Main Function ---

def main():
    st.title("Agent-as-Coder: PDF Parser Generator")
    st.markdown("""
    This app demonstrates an autonomous agent that generates a Python parser for bank statement PDFs.
    
    **Instructions:**
    1.  Upload a sample PDF statement and a corresponding CSV file with the expected data.
    2.  Provide a name for the target bank.
    3.  Click "Run Agent" to start the process.
    
   
    """)
    
    # --- File Uploads ---
    pdf_file = st.file_uploader("Upload PDF Statement", type="pdf")
    csv_file = st.file_uploader("Upload Expected CSV", type="csv")
    target_bank = st.text_input("Enter Target Bank Name (e.g., icici)", "icici")

    # --- Run Button ---
    if st.button("Run Agent"):
        if not pdf_file or not csv_file or not target_bank:
            st.error("Please upload both a PDF and a CSV, and provide a target bank name.")
            return

        if not os.getenv("GROQ_API_KEY"):
            st.error("Error: The GROQ_API_KEY environment variable is not set. Please set it to proceed.")
            return

        st.header("Agent Execution Log")
        
        # Save uploaded files to a temporary location
        data_dir = "./temp_data"
        os.makedirs(data_dir, exist_ok=True)
        pdf_path = os.path.join(data_dir, pdf_file.name)
        csv_path = os.path.join(data_dir, csv_file.name)
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        with open(csv_path, "wb") as f:
            f.write(csv_file.getbuffer())
        
        # Load CSV to get the schema
        try:
            df = pd.read_csv(csv_path)
            csv_schema = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'shape': df.shape
            }
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return
            
        # Initialize the agent state
        initial_state = AgentState(
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
        
        # Create and run workflow
        workflow = create_workflow()
        app = workflow.compile()
        
        st.write(f"🚀 Starting agent for {target_bank} bank...")
        st.write(f"📄 PDF: {pdf_file.name}")
        st.write(f"📊 CSV: {csv_file.name}")
        st.write(f"🏗️ Schema: {csv_schema['shape']} with columns: {csv_schema['columns']}")
        st.write("-" * 50)
        
        try:
            final_state = app.invoke(initial_state)
            
            if final_state.final_parser_path:
                st.balloons()
                st.success(f"\n🎉 Success! Parser generated at: {final_state.final_parser_path}")
                st.write("Final Code:")
                st.code(final_state.generated_code, language="python")
            else:
                st.error(f"\n❌ Failed after {final_state.attempts} attempts")
                st.write("Errors:")
                for error in final_state.errors:
                    st.error(error)
        
        except Exception as e:
            st.error(f"❌ Workflow failed: {e}")

if __name__ == "__main__":
    main()
