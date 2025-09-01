#!/usr/bin/env python3
"""
Streamlit App for ICICI Bank Statement Analysis Demo
A comprehensive demonstration of PDF processing, data extraction, and analytics
"""

import streamlit as st
import pandas as pd
import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ICICI Bank Statement Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the expected result CSV and PDF data"""
    try:
        result_csv_path = "data/icici/result.csv"
        pdf_path = "data/icici/icici sample.pdf"
        
        if Path(result_csv_path).exists():
            expected_df = pd.read_csv(result_csv_path)
        else:
            st.error(f"Expected CSV file not found: {result_csv_path}")
            return None, None, None
            
        if Path(pdf_path).exists():
            return expected_df, pdf_path, None
        else:
            st.error(f"PDF file not found: {pdf_path}")
            return expected_df, None, None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def extract_transactions_from_pdf(pdf_path):
    """Extract transaction data from ICICI bank statement PDF"""
    transactions = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                        
                    # Look for transaction tables (should have multiple columns)
                    if len(table[0]) >= 4:
                        # Process each row (skip header)
                        for row in table[1:]:
                            if len(row) >= 4 and any(cell and str(cell).strip() for cell in row):
                                # Extract transaction data
                                transaction = {
                                    'Date': row[0] if len(row) > 0 else None,
                                    'Description': row[1] if len(row) > 1 else None,
                                    'Debit Amt': row[2] if len(row) > 2 else None,
                                    'Credit Amt': row[3] if len(row) > 3 else None,
                                    'Balance': row[4] if len(row) > 4 else None
                                }
                                
                                # Clean up the data
                                for key, value in transaction.items():
                                    if value is not None:
                                        # Remove extra whitespace
                                        if isinstance(value, str):
                                            value = value.strip()
                                        # Convert empty strings to None
                                        if value == '':
                                            value = None
                                        transaction[key] = value
                                
                                # Only add if we have meaningful data
                                if transaction['Date'] and transaction['Description']:
                                    transactions.append(transaction)
    except Exception as e:
        st.error(f"Error extracting from PDF: {e}")
        return []
    
    return transactions

def clean_and_format_transactions(df):
    """Clean and format the extracted transaction data"""
    if df.empty:
        return df
        
    cleaned_df = df.copy()
    
    # Convert numeric columns
    numeric_columns = ['Debit Amt', 'Credit Amt', 'Balance']
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove currency symbols and commas
            cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r'[₹,₹\s]', '', regex=True)
            # Convert to numeric, errors='coerce' will set invalid values to NaN
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Clean date column
    if 'Date' in cleaned_df.columns:
        cleaned_df['Date'] = cleaned_df['Date'].astype(str).str.strip()
        cleaned_df['Date'] = cleaned_df['Date'].replace('', None)
    
    # Clean description column
    if 'Description' in cleaned_df.columns:
        cleaned_df['Description'] = cleaned_df['Description'].astype(str).str.strip()
        cleaned_df['Description'] = cleaned_df['Description'].replace('', None)
    
    # Remove rows where both Date and Description are None
    cleaned_df = cleaned_df.dropna(subset=['Date', 'Description'], how='all')
    
    return cleaned_df

def validate_schema(extracted_df, expected_df):
    """Validate that the extracted data matches the expected schema"""
    if extracted_df.empty or expected_df.empty:
        return False, "Empty DataFrames"
    
    # Check column names
    expected_columns = list(expected_df.columns)
    extracted_columns = list(extracted_df.columns)
    
    missing_columns = set(expected_columns) - set(extracted_columns)
    extra_columns = set(extracted_columns) - set(expected_columns)
    
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    return True, "Schema validation passed"

def create_transaction_distribution_chart(df):
    """Create transaction distribution chart"""
    if df.empty:
        return None
        
    # Count debit vs credit transactions
    debit_count = (df['Debit Amt'] > 0).sum() if 'Debit Amt' in df.columns else 0
    credit_count = (df['Credit Amt'] > 0).sum() if 'Credit Amt' in df.columns else 0
    
    fig = px.pie(
        values=[debit_count, credit_count],
        names=['Debit Transactions', 'Credit Transactions'],
        title='Transaction Distribution',
        color_discrete_sequence=['#ff6b6b', '#4ecdc4']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_amount_distribution_chart(df):
    """Create amount distribution histogram"""
    if df.empty or 'Debit Amt' not in df.columns:
        return None
        
    debit_amounts = df['Debit Amt'].dropna()
    if len(debit_amounts) == 0:
        return None
        
    fig = px.histogram(
        x=debit_amounts,
        title='Debit Amount Distribution',
        nbins=20,
        color_discrete_sequence=['#ff6b6b']
    )
    fig.update_layout(xaxis_title='Amount', yaxis_title='Frequency')
    return fig

def create_balance_trend_chart(df):
    """Create balance trend chart"""
    if df.empty or 'Balance' not in df.columns:
        return None
        
    balance_data = df['Balance'].dropna()
    if len(balance_data) == 0:
        return None
        
    fig = px.line(
        x=range(len(balance_data)),
        y=balance_data,
        title='Balance Over Transactions',
        markers=True,
        color_discrete_sequence=['#2e8b57']
    )
    fig.update_layout(
        xaxis_title='Transaction Number',
        yaxis_title='Balance'
    )
    return fig

def create_data_completeness_chart(df):
    """Create data completeness chart"""
    if df.empty:
        return None
        
    completeness = df.notna().sum() / len(df) * 100
    
    fig = px.bar(
        x=completeness.index,
        y=completeness.values,
        title='Data Completeness by Column',
        color_discrete_sequence=['#87ceeb']
    )
    fig.update_layout(
        xaxis_title='Columns',
        yaxis_title='Completeness (%)',
        yaxis_range=[0, 100]
    )
    
    # Add percentage labels on bars
    for i, v in enumerate(completeness.values):
        fig.add_annotation(
            x=i, y=v + 2,
            text=f'{v:.1f}%',
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig

def cleanup_temp_files():
    """Clean up temporary files created during the session"""
    import tempfile
    import os
    
    # Clean up any temporary PDF files
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith('tmp') and file.endswith('.pdf'):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass

def main():
    """Main Streamlit application"""
    
    # Register cleanup function
    import atexit
    atexit.register(cleanup_temp_files)
    
    # Header
    st.markdown('<h1 class="main-header">🏦 ICICI Bank Statement Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Demo & Analytics Dashboard")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📄 PDF Analysis", "🔍 Data Extraction", "📊 Analytics", "🧪 Parser Testing", "📈 Visualizations"]
    )
    
    # PDF Upload Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Upload Your Own PDF")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload your own bank statement PDF for analysis"
    )
    
    # Load data
    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
            
        # For uploaded files, we'll use a generic expected schema
        expected_df = pd.DataFrame({
            'Date': [],
            'Description': [],
            'Debit Amt': [],
            'Credit Amt': [],
            'Balance': []
        })
    else:
        # Use default data
        expected_df, pdf_path, _ = load_data()
        
        if expected_df is None:
            st.error("❌ Failed to load data. Please check your file paths.")
            return
    
    # File Information Section
    if uploaded_file is not None:
        st.markdown('<h2 class="section-header">📤 Uploaded File Information</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        st.info("💡 This is your uploaded PDF. The app will analyze it and extract transaction data.")
        st.markdown("---")
    
    # Overview Page
    if page == "🏠 Overview":
        st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What This App Demonstrates
            
            This Streamlit application showcases the complete workflow of processing ICICI bank statements:
            
            - **PDF Analysis**: Understanding document structure
            - **Data Extraction**: Using pdfplumber to extract tables
            - **Data Processing**: Cleaning and formatting raw data
            - **Schema Validation**: Ensuring output matches expected format
            - **Analytics**: Comprehensive data analysis and insights
            - **Visualizations**: Interactive charts and graphs
            
            ### 🚀 Key Features
            
            - Interactive PDF processing
            - Real-time data extraction
            - Automated schema validation
            - Comprehensive analytics dashboard
            - Beautiful visualizations
            - Parser testing capabilities
            """)
        
        with col2:
            st.markdown("""
            ### 📁 Data Files
            
            - **Input**: `icici sample.pdf` - ICICI bank statement
            - **Expected Output**: `result.csv` - Reference data format
            - **Parser**: `icici_parser.py` - Custom extraction logic
            
            ### 🔧 Technology Stack
            
            - **Backend**: Python, Pandas, pdfplumber
            - **Frontend**: Streamlit
            - **Visualization**: Plotly, Matplotlib, Seaborn
            - **AI Agent**: LangGraph, Groq AI
            
            ### 📊 Expected Schema
            
            The system expects data with these columns:
            - Date
            - Description  
            - Debit Amt
            - Credit Amt
            - Balance
            """)
        
        # Metrics
        st.markdown('<h3 class="section-header">📈 Key Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expected Rows", len(expected_df))
        
        with col2:
            st.metric("Expected Columns", len(expected_df.columns))
        
        with col3:
            st.metric("Data Types", len(expected_df.dtypes.unique()))
        
        with col4:
            st.metric("File Size", f"{Path('data/icici/result.csv').stat().st_size / 1024:.1f} KB")
        
        # Show expected data preview
        st.markdown('<h3 class="section-header">📋 Expected Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(expected_df.head(10), use_container_width=True)
    
    # PDF Analysis Page
    elif page == "📄 PDF Analysis":
        st.markdown('<h2 class="section-header">PDF Structure Analysis</h2>', unsafe_allow_html=True)
        
        if pdf_path:
            st.info(f"📄 Analyzing PDF: {pdf_path}")
            
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    st.success(f"✅ PDF loaded successfully! Total pages: {len(pdf.pages)}")
                    
                    # Page-by-page analysis
                    for page_num, page in enumerate(pdf.pages):
                        st.markdown(f"### 📖 Page {page_num + 1}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Extract text
                            text = page.extract_text()
                            if text:
                                st.metric("Text Length", f"{len(text)} characters")
                                st.text_area(f"Page {page_num + 1} Text (First 500 chars)", 
                                           text[:500] + "..." if len(text) > 500 else text, 
                                           height=150)
                        
                        with col2:
                            # Extract tables
                            tables = page.extract_tables()
                            st.metric("Tables Found", len(tables))
                            
                            if tables:
                                for table_idx, table in enumerate(tables):
                                    if table and len(table) > 0:
                                        st.markdown(f"**Table {table_idx + 1}:** {len(table)} rows × {len(table[0]) if table[0] else 0} columns")
                                        
                                        # Show table preview
                                        if len(table) > 0:
                                            df_preview = pd.DataFrame(table[:5])  # Show first 5 rows
                                            st.dataframe(df_preview, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error analyzing PDF: {e}")
        else:
            st.error("❌ PDF file not found")
    
    # Data Extraction Page
    elif page == "🔍 Data Extraction":
        st.markdown('<h2 class="section-header">Data Extraction & Processing</h2>', unsafe_allow_html=True)
        
        if pdf_path:
            st.info("🔍 Extracting transaction data from PDF...")
            
            # Extract transactions
            with st.spinner("Processing PDF..."):
                extracted_transactions = extract_transactions_from_pdf(pdf_path)
            
            if extracted_transactions:
                st.success(f"✅ Successfully extracted {len(extracted_transactions)} transactions!")
                
                # Convert to DataFrame
                extracted_df = pd.DataFrame(extracted_transactions)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Raw Extracted Data")
                    st.dataframe(extracted_df.head(10), use_container_width=True)
                    st.info(f"Shape: {extracted_df.shape}")
                
                with col2:
                    st.markdown("### 📋 Column Information")
                    for col in extracted_df.columns:
                        non_null_count = extracted_df[col].notna().sum()
                        st.metric(f"{col}", f"{non_null_count}/{len(extracted_df)} ({non_null_count/len(extracted_df)*100:.1f}%)")
                
                # Data cleaning
                st.markdown("### 🧹 Data Cleaning & Formatting")
                
                with st.spinner("Cleaning data..."):
                    cleaned_df = clean_and_format_transactions(extracted_df)
                
                if not cleaned_df.empty:
                    st.success("✅ Data cleaned successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Before Cleaning:**")
                        st.dataframe(extracted_df.head(5), use_container_width=True)
                    
                    with col2:
                        st.markdown("**After Cleaning:**")
                        st.dataframe(cleaned_df.head(5), use_container_width=True)
                    
                    # Schema validation
                    st.markdown("### ✅ Schema Validation")
                    is_valid, message = validate_schema(cleaned_df, expected_df)
                    
                    if is_valid:
                        st.success(f"✅ {message}")
                        
                        # Show comparison
                        st.markdown("### 📊 Data Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Expected Data (result.csv):**")
                            st.dataframe(expected_df.head(5), use_container_width=True)
                        
                        with col2:
                            st.markdown("**Extracted Data (from PDF):**")
                            st.dataframe(cleaned_df.head(5), use_container_width=True)
                        
                        # Check for exact matches
                        if cleaned_df.shape == expected_df.shape:
                            st.success("✅ Dimensions match perfectly!")
                        else:
                            st.warning(f"⚠️ Dimension mismatch: Expected {expected_df.shape}, Got {cleaned_df.shape}")
                    else:
                        st.error(f"❌ Schema validation failed: {message}")
                else:
                    st.error("❌ Data cleaning failed - no data remaining")
            else:
                st.error("❌ No transactions extracted from PDF")
        else:
            st.error("❌ PDF file not found")
    
    # Analytics Page
    elif page == "📊 Analytics":
        st.markdown('<h2 class="section-header">Data Analytics & Insights</h2>', unsafe_allow_html=True)
        
        if pdf_path:
            # Extract and process data for analytics
            with st.spinner("Preparing analytics..."):
                extracted_transactions = extract_transactions_from_pdf(pdf_path)
                if extracted_transactions:
                    extracted_df = pd.DataFrame(extracted_transactions)
                    cleaned_df = clean_and_format_transactions(extracted_df)
                else:
                    st.error("❌ No data available for analytics")
                    return
            
            if not cleaned_df.empty:
                st.success("✅ Analytics data ready!")
                
                # Key metrics
                st.markdown("### 📈 Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", len(cleaned_df))
                
                with col2:
                    if 'Debit Amt' in cleaned_df.columns:
                        debit_count = (cleaned_df['Debit Amt'] > 0).sum()
                        st.metric("Debit Transactions", debit_count)
                
                with col3:
                    if 'Credit Amt' in cleaned_df.columns:
                        credit_count = (cleaned_df['Credit Amt'] > 0).sum()
                        st.metric("Credit Transactions", credit_count)
                
                with col4:
                    if 'Balance' in cleaned_df.columns:
                        avg_balance = cleaned_df['Balance'].mean()
                        st.metric("Average Balance", f"₹{avg_balance:,.2f}" if not pd.isna(avg_balance) else "N/A")
                
                # Data quality metrics
                st.markdown("### 🔍 Data Quality Metrics")
                
                quality_data = []
                for col in cleaned_df.columns:
                    completeness = cleaned_df[col].notna().sum() / len(cleaned_df) * 100
                    quality_data.append({
                        'Column': col,
                        'Completeness (%)': completeness,
                        'Missing Values': cleaned_df[col].isna().sum(),
                        'Unique Values': cleaned_df[col].nunique()
                    })
                
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df, use_container_width=True)
                
                # Statistical summary
                st.markdown("### 📊 Statistical Summary")
                
                if 'Debit Amt' in cleaned_df.columns and 'Credit Amt' in cleaned_df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Debit Amount Statistics:**")
                        debit_stats = cleaned_df['Debit Amt'].describe()
                        st.dataframe(pd.DataFrame(debit_stats).T, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Credit Amount Statistics:**")
                        credit_stats = cleaned_df['Credit Amt'].describe()
                        st.dataframe(pd.DataFrame(credit_stats).T, use_container_width=True)
                
                # Balance analysis
                if 'Balance' in cleaned_df.columns:
                    st.markdown("### 💰 Balance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Balance Statistics:**")
                        balance_stats = cleaned_df['Balance'].describe()
                        st.dataframe(pd.DataFrame(balance_stats).T, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Balance Range:**")
                        min_balance = cleaned_df['Balance'].min()
                        max_balance = cleaned_df['Balance'].max()
                        st.metric("Minimum Balance", f"₹{min_balance:,.2f}" if not pd.isna(min_balance) else "N/A")
                        st.metric("Maximum Balance", f"₹{max_balance:,.2f}" if not pd.isna(max_balance) else "N/A")
            else:
                st.error("❌ No cleaned data available for analytics")
        else:
            st.error("❌ PDF file not found")
    
    # Parser Testing Page
    elif page == "🧪 Parser Testing":
        st.markdown('<h2 class="section-header">Custom Parser Testing</h2>', unsafe_allow_html=True)
        
        st.info("🧪 Testing our custom ICICI parser implementation")
        
        try:
            # Import custom parser
            import sys
            sys.path.append('.')
            
            from custom_parsers.icici_parser import parse
            
            st.success("✅ Custom parser imported successfully!")
            
            if pdf_path:
                # Test parser
                with st.spinner("Running parser..."):
                    try:
                        parser_result = parse(pdf_path)
                        st.success("✅ Parser executed successfully!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Parser Output")
                            st.dataframe(parser_result.head(10), use_container_width=True)
                            st.info(f"Shape: {parser_result.shape}")
                        
                        with col2:
                            st.markdown("### 📋 Parser Columns")
                            for col in parser_result.columns:
                                non_null_count = parser_result[col].notna().sum()
                                st.metric(f"{col}", f"{non_null_count}/{len(parser_result)} ({non_null_count/len(parser_result)*100:.1f}%)")
                        
                        # Compare with expected output
                        st.markdown("### 🔍 Parser Validation")
                        
                        if parser_result.shape == expected_df.shape:
                            st.success("✅ Parser output matches expected dimensions!")
                            
                            # Check column names
                            if set(parser_result.columns) == set(expected_df.columns):
                                st.success("✅ Column names match expected schema!")
                            else:
                                st.warning("⚠️ Column names don't match expected schema")
                                st.markdown(f"**Expected:** {list(expected_df.columns)}")
                                st.markdown(f"**Got:** {list(parser_result.columns)}")
                        else:
                            st.warning(f"⚠️ Dimension mismatch: Expected {expected_df.shape}, Got {parser_result.shape}")
                        
                        # Performance metrics
                        st.markdown("### ⚡ Performance Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Rows Processed", len(parser_result))
                        
                        with col2:
                            st.metric("Columns Generated", len(parser_result.columns))
                        
                        with col3:
                            st.metric("Data Completeness", f"{parser_result.notna().sum().sum() / (len(parser_result) * len(parser_result.columns)) * 100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Parser execution failed: {e}")
                        st.info("💡 This might be expected if the parser needs refinement")
            else:
                st.error("❌ PDF file not found")
                
        except ImportError as e:
            st.error(f"❌ Could not import custom parser: {e}")
            st.info("💡 Make sure the custom_parsers directory is in your Python path")
    
    # Visualizations Page
    elif page == "📈 Visualizations":
        st.markdown('<h2 class="section-header">Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        if pdf_path:
            # Extract and process data for visualizations
            with st.spinner("Preparing visualizations..."):
                extracted_transactions = extract_transactions_from_pdf(pdf_path)
                if extracted_transactions:
                    extracted_df = pd.DataFrame(extracted_transactions)
                    cleaned_df = clean_and_format_transactions(extracted_df)
                else:
                    st.error("❌ No data available for visualizations")
                    return
            
            if not cleaned_df.empty:
                st.success("✅ Visualization data ready!")
                
                # Transaction distribution
                st.markdown("### 🥧 Transaction Distribution")
                dist_fig = create_transaction_distribution_chart(cleaned_df)
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for transaction distribution chart")
                
                # Amount distribution
                st.markdown("### 📊 Amount Distribution")
                amount_fig = create_amount_distribution_chart(cleaned_df)
                if amount_fig:
                    st.plotly_chart(amount_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for amount distribution chart")
                
                # Balance trend
                st.markdown("### 📈 Balance Trend")
                balance_fig = create_balance_trend_chart(cleaned_df)
                if balance_fig:
                    st.plotly_chart(balance_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for balance trend chart")
                
                # Data completeness
                st.markdown("### 📋 Data Completeness")
                completeness_fig = create_data_completeness_chart(cleaned_df)
                if completeness_fig:
                    st.plotly_chart(completeness_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data for completeness chart")
                
                # Interactive data explorer
                st.markdown("### 🔍 Interactive Data Explorer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_column = st.selectbox("Select column to analyze:", cleaned_df.columns)
                
                with col2:
                    if selected_column in cleaned_df.columns:
                        if cleaned_df[selected_column].dtype in ['int64', 'float64']:
                            chart_type = st.selectbox("Chart type:", ["Histogram", "Box Plot", "Line Chart"])
                        else:
                            chart_type = st.selectbox("Chart type:", ["Bar Chart", "Value Counts"])
                
                # Generate selected chart
                if selected_column in cleaned_df.columns:
                    if chart_type == "Histogram" and cleaned_df[selected_column].dtype in ['int64', 'float64']:
                        fig = px.histogram(cleaned_df, x=selected_column, title=f'{selected_column} Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Box Plot" and cleaned_df[selected_column].dtype in ['int64', 'float64']:
                        fig = px.box(cleaned_df, y=selected_column, title=f'{selected_column} Box Plot')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Line Chart" and cleaned_df[selected_column].dtype in ['int64', 'float64']:
                        fig = px.line(cleaned_df, y=selected_column, title=f'{selected_column} Trend')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Bar Chart":
                        value_counts = cleaned_df[selected_column].value_counts().head(20)
                        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f'{selected_column} Value Counts')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Value Counts":
                        st.markdown(f"**{selected_column} Value Counts:**")
                        value_counts = cleaned_df[selected_column].value_counts()
                        st.dataframe(value_counts.head(20), use_container_width=True)
            else:
                st.error("❌ No cleaned data available for visualizations")
        else:
            st.error("❌ PDF file not found")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🏦 <strong>ICICI Bank Statement Analyzer</strong> | Built with Streamlit & Python</p>
        <p>Demonstrating AI-powered PDF processing and financial data analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 