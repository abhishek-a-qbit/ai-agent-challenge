#!/usr/bin/env python3
"""
Test file for the Agent-as-Coder challenge
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_csv_schema():
    """Test that sample CSV can be loaded and has expected structure"""
    csv_path = "data/icici/icici_sample.csv"
    assert Path(csv_path).exists(), f"CSV file not found: {csv_path}"
    
    df = pd.read_csv(csv_path)
    expected_columns = ['Date', 'Description', 'Debit', 'Credit', 'Balance']
    
    assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"
    assert len(df) > 0, "CSV should contain data"
    assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"

def test_agent_import():
    """Test that agent.py can be imported"""
    try:
        import agent
        assert hasattr(agent, 'main'), "agent.py should have main function"
        assert hasattr(agent, 'AgentState'), "agent.py should have AgentState class"
        print("✅ agent.py imports successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import agent.py: {e}")

def test_parser_template():
    """Test that parser template exists and has correct structure"""
    parser_path = "custom_parsers/icici_parser.py"
    assert Path(parser_path).exists(), f"Parser template not found: {parser_path}"
    
    # Check that parse function exists
    with open(parser_path, 'r') as f:
        content = f.read()
        assert 'def parse(' in content, "Parser should have parse function"
        assert 'pd.DataFrame' in content, "Parser should return DataFrame"
        assert 'pdfplumber' in content, "Parser should use pdfplumber"

def test_project_structure():
    """Test that project has required structure"""
    required_files = [
        'agent.py',
        'requirements.txt',
        'README.md',
        'custom_parsers/__init__.py',
        'custom_parsers/icici_parser.py',
        'data/icici/icici_sample.csv'
    ]
    
    for file_path in required_files:
        assert Path(file_path).exists(), f"Required file not found: {file_path}"
    
    print("✅ All required files present")

if __name__ == "__main__":
    print("🧪 Running Agent-as-Coder tests...")
    
    test_csv_schema()
    test_agent_import()
    test_parser_template()
    test_project_structure()
    
    print("🎉 All tests passed!") 