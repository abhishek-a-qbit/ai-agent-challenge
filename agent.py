#!/usr/bin/env python3
"""
Agent-as-Coder: PDF Parser Generator Agent

This agent can generate custom parsers for bank statement PDFs by analyzing
sample PDFs and their corresponding CSV outputs.
"""

import os
import json
import click
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import dotenv

# Load environment variables
dotenv.load_dotenv()

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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF structure and return key insights"""
        try:
            import pdfplumber
            
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
                First page text preview: {pages[0]['text'][:500] if pages else 'No text'}
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
            model="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def generate_parser(self, pdf_analysis: Dict[str, Any], csv_schema: Dict[str, Any], target_bank: str) -> str:
        """Generate parser code based on PDF analysis and CSV schema"""
        
        prompt = f"""
        Generate a Python parser for {target_bank} bank statements.
        
        Requirements:
        1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
        2. Must return DataFrame matching this schema: {csv_schema}
        3. Use pdfplumber for PDF processing
        4. Handle errors gracefully
        5. Include proper type hints and docstring
        
        PDF Analysis: {pdf_analysis.get('analysis', 'No analysis available')}
        
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
            import importlib.util
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
    print(f"🔍 Analyzing PDF: {state.pdf_path}")
    
    analyzer = PDFAnalyzer()
    analysis = analyzer.analyze_pdf_structure(state.pdf_path)
    
    if 'error' in analysis:
        state.errors.append(f"PDF analysis failed: {analysis['error']}")
        return state
    
    # Store analysis in state for code generation
    state.pdf_analysis = analysis
    print("✅ PDF analysis completed")
    return state

def generate_code(state: AgentState) -> AgentState:
    """Node: Generate parser code"""
    print(f"💻 Generating parser code for {state.target_bank}")
    
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
    print("✅ Code generation completed")
    return state

def test_code(state: AgentState) -> AgentState:
    """Node: Test generated code"""
    print("🧪 Testing generated code...")
    
    tester = CodeTester()
    test_results = tester.test_parser(
        state.generated_code,
        state.pdf_path,
        state.csv_path
    )
    
    state.test_results = test_results
    
    if test_results['success'] and test_results['is_equal']:
        print("✅ Code test passed!")
        return state
    else:
        state.errors.append(f"Test failed: {test_results.get('error', 'DataFrame mismatch')}")
        print("❌ Code test failed")
        return state

def save_parser(state: AgentState) -> AgentState:
    """Node: Save the working parser"""
    print("💾 Saving parser...")
    
    # Create custom_parsers directory if it doesn't exist
    parser_dir = Path("custom_parsers")
    parser_dir.mkdir(exist_ok=True)
    
    # Save parser
    parser_path = parser_dir / f"{state.target_bank}_parser.py"
    with open(parser_path, 'w') as f:
        f.write(state.generated_code)
    
    state.final_parser_path = str(parser_path)
    print(f"✅ Parser saved to: {parser_path}")
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
    print(f"🔄 Retry attempt {state.attempts}/3")
    
    # Generate improved code based on test feedback
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
    
    # Add nodes
    workflow.add_node("analyze_pdf", analyze_pdf)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("test_code", test_code)
    workflow.add_node("save_parser", save_parser)
    workflow.add_node("retry", retry_with_feedback)
    
    # Add edges
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

@click.command()
@click.option('--target', required=True, help='Target bank (e.g., icici)')
@click.option('--pdf', help='Path to PDF file (default: data/{target}/{target}_sample.pdf)')
@click.option('--csv', help='Path to CSV file (default: data/{target}/{target}_sample.csv)')
def main(target: str, pdf: str = None, csv: str = None):
    """Generate custom PDF parser for bank statements"""
    
    # Set default paths if not provided
    if not pdf:
        pdf = f"data/{target}/{target}_sample.pdf"
    if not csv:
        csv = f"data/{target}/{target}_sample.csv"
    
    # Check if files exist
    if not os.path.exists(pdf):
        click.echo(f"❌ PDF file not found: {pdf}")
        return
    if not os.path.exists(csv):
        click.echo(f"❌ CSV file not found: {csv}")
        return
    
    # Load CSV to understand schema
    try:
        df = pd.read_csv(csv)
        csv_schema = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape
        }
    except Exception as e:
        click.echo(f"❌ Failed to read CSV: {e}")
        return
    
    # Initialize state
    initial_state = AgentState(
        target_bank=target,
        pdf_path=pdf,
        csv_path=csv,
        csv_schema=csv_schema,
        generated_code="",
        test_results={},
        attempts=0,
        errors=[],
        final_parser_path=""
    )
    
    # Create and run workflow
    workflow = create_workflow()
    app = workflow.compile(checkpointer=MemorySaver())
    
    print(f"🚀 Starting agent for {target} bank...")
    print(f"📄 PDF: {pdf}")
    print(f"📊 CSV: {csv}")
    print(f"🏗️  Schema: {csv_schema['shape']} with columns: {csv_schema['columns']}")
    print("-" * 50)
    
    try:
        result = app.invoke(initial_state)
        
        if result.final_parser_path:
            print(f"\n🎉 Success! Parser generated at: {result.final_parser_path}")
            print(f"📊 Final test results: {result.test_results}")
        else:
            print(f"\n❌ Failed after {result.attempts} attempts")
            print(f"Errors: {result.errors}")
            
    except Exception as e:
        print(f"❌ Workflow failed: {e}")

if __name__ == "__main__":
    main() 