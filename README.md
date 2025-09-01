# Agent-as-Coder: PDF Parser Generator

An intelligent coding agent that automatically generates custom parsers for bank statement PDFs using LangGraph and Groq AI.

## 🎯 Challenge Overview

This project implements an "Agent-as-Coder" that can:
- Analyze PDF structure using AI
- Generate custom Python parsers
- Test and validate generated code
- Self-correct through multiple attempts (≤3)
- Handle different bank statement formats

## 🏗️ Agent Architecture

The agent follows a **LangGraph-based workflow** with the following components:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  PDF Input │───▶│   Analyze    │───▶│   Generate  │───▶│    Test     │
│  + CSV     │    │   Structure  │    │    Code     │    │   Parser    │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                                              │
                    ┌─────────────┐    ┌─────────────┐       │
                    │    Save     │◀───│   Retry     │◀──────┘
                    │   Parser    │    │  (≤3x)     │
                    └─────────────┘    └─────────────┘
```

**Key Features:**
- **Autonomous Operation**: Self-debugging loops with intelligent error handling
- **Multi-Attempt Generation**: Up to 3 attempts with feedback-based improvement
- **Schema Validation**: Ensures output matches expected CSV structure
- **Modular Design**: Clean separation of concerns (Analysis, Generation, Testing)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/abhishek-a-qbit/ai-agent-challenge.git
cd ai-agent-challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get free API credits from:
- [Groq](https://console.groq.com/)

### 3. Run the Agent

```bash
# Generate parser for ICICI bank
python agent.py --target icici

# Use custom file paths
python agent.py --target sbi --pdf data/sbi/statement.pdf --csv data/sbi/expected.csv
```

### 4. Test Generated Parser

```bash
# Run tests
pytest

# Test specific parser
python custom_parsers/icici_parser.py data/icici/icici_sample.pdf
```

## 📁 Project Structure

```
ai-agent-challenge/
├── agent.py                 # Main agent implementation
├── requirements.txt         # Python dependencies
├── README.md              # This file
├── .env                   # API keys (create this)
├── custom_parsers/        # Generated parsers
│   ├── __init__.py
│   └── icici_parser.py    # Sample parser
└── data/                  # Sample data
    └── icici/
        ├── icici sample.pdf
        └── result.csv
```

## 🔧 Core Components

### 1. PDFAnalyzer
- Extracts text and table structures from PDFs
- Uses AI to analyze document structure and patterns
- Identifies key sections and data formats

### 2. CodeGenerator
- Generates Python parser code based on PDF analysis
- Ensures code matches expected CSV schema
- Includes proper error handling and type hints

### 3. CodeTester
- Dynamically tests generated code
- Compares output with expected CSV data
- Provides detailed feedback for improvements

### 4. Workflow Orchestrator
- Manages the agent's decision flow
- Handles retry logic (≤3 attempts)
- Routes between analysis, generation, testing, and saving

## 📊 Parser Contract

All generated parsers implement this interface:

```python
def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse bank statement PDF and return transaction data.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        DataFrame matching expected CSV schema
    """
    pass
```

## 🧪 Testing

The agent automatically tests generated parsers:

```bash
# Run all tests
pytest

# Test specific functionality
python -m pytest tests/ -v
```

## 🎭 Demo Requirements

**60-second demo checklist:**
1. ✅ Fresh clone of repository
2. ✅ Run `python agent.py --target icici`
3. ✅ Show green pytest results
4. ✅ Demonstrate generated parser working

## 🔍 Troubleshooting

### Common Issues:

1. **API Key Error**: Ensure `.env` file exists with `GOOGLE_API_KEY`
2. **PDF Not Found**: Check file paths in `data/` directory
3. **Import Errors**: Verify all dependencies are installed
4. **Memory Issues**: Large PDFs may require more RAM

### Debug Mode:

```bash
# Enable verbose logging
export PYTHONPATH=.
python agent.py --target icici --debug
```

## 🚀 Future Enhancements

- Support for more document types (Excel, images)
- Advanced error recovery mechanisms
- Parallel processing for large documents
- Integration with more LLM providers
- Web interface for non-technical users

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the AI Agent Challenge. See original repository for license details.

---

**Happy Hacking! 🎉**

Remember: Keep it simple, commit often, and let the agent do the heavy lifting!
