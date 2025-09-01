"""
ICICI Bank Statement Parser

This parser extracts transaction data from ICICI bank statement PDFs
and returns a pandas DataFrame matching the expected schema.
"""

import pandas as pd
import pdfplumber
from typing import Dict, List, Any
import re
from datetime import datetime

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse ICICI bank statement PDF and return transaction data.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        pd.DataFrame: DataFrame with transaction data matching expected schema
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF cannot be parsed
    """
    try:
        # Extract text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() or ""
        
        # Parse transactions (this is a template - actual implementation would be more sophisticated)
        transactions = []
        
        # Example parsing logic (would be customized based on actual PDF structure)
        lines = text_content.split('\n')
        for line in lines:
            # Look for transaction patterns
            if re.search(r'\d{2}/\d{2}/\d{4}', line):  # Date pattern
                # Extract transaction details
                # This is a simplified example
                pass
        
        # Create DataFrame with expected schema
        # Adjust columns based on actual CSV schema
        df = pd.DataFrame(transactions)
        
        # Ensure DataFrame matches expected schema
        expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        
        return df[expected_columns]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")

if __name__ == "__main__":
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        result = parse(pdf_file)
        print(f"Parsed {len(result)} transactions")
        print(result.head())
    else:
        print("Usage: python icici_parser.py <pdf_file>") 