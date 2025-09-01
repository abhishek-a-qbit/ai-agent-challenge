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
        # Extract tables from PDF
        with pdfplumber.open(pdf_path) as pdf:
            all_tables = []
            
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
            
            if not all_tables:
                raise ValueError("No tables found in PDF")
            
            # Process tables to extract transaction data
            transactions = []
            
            for table in all_tables:
                if not table or len(table) < 2:  # Skip empty tables or tables without headers
                    continue
                
                # Check if this table has the expected structure
                headers = table[0]
                if len(headers) >= 5 and 'Date' in str(headers[0]):
                    # Process transaction rows
                    for row in table[1:]:  # Skip header row
                        if len(row) >= 5 and row[0]:  # Ensure row has data
                            transaction = {
                                'Date': row[0] if row[0] else None,
                                'Description': row[1] if row[1] else None,
                                'Debit Amt': row[2] if row[2] else None,
                                'Credit Amt': row[3] if row[3] else None,
                                'Balance': row[4] if row[4] else None
                            }
                            transactions.append(transaction)
            
            # Create DataFrame
            df = pd.DataFrame(transactions)
            
            # Ensure DataFrame matches expected schema
            expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Clean and format data
            df = df[expected_columns]  # Ensure correct column order
            
            # Remove empty rows
            df = df.dropna(subset=['Date']).reset_index(drop=True)
            
            # Convert numeric columns
            numeric_columns = ['Debit Amt', 'Credit Amt', 'Balance']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
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
        print(f"\nColumns: {list(result.columns)}")
        print(f"Shape: {result.shape}")
    else:
        print("Usage: python icici_parser.py <pdf_file>") 