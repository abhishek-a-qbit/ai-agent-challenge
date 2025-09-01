# 🏦 ICICI Bank Statement Analyzer - Streamlit App

A comprehensive, interactive Streamlit application that demonstrates the complete workflow of processing ICICI bank statements, extracting transaction data, and performing advanced analytics.

## 🎯 What This App Demonstrates

This Streamlit application showcases the complete workflow of our AI-powered bank statement processing system:

- **📄 PDF Analysis**: Understanding document structure and extracting tables
- **🔍 Data Extraction**: Using pdfplumber to extract transaction data
- **🧹 Data Processing**: Cleaning and formatting raw data
- **✅ Schema Validation**: Ensuring output matches expected CSV structure
- **📊 Analytics**: Comprehensive data analysis and insights
- **📈 Visualizations**: Interactive charts and graphs using Plotly
- **🧪 Parser Testing**: Testing our custom parser implementation

## 🚀 Key Features

### 🏠 Overview Dashboard
- Project overview and technology stack
- Key metrics and data file information
- Expected schema documentation
- Interactive navigation

### 📄 PDF Analysis
- Page-by-page PDF structure analysis
- Table extraction and preview
- Text content analysis
- Real-time processing feedback

### 🔍 Data Extraction
- Live PDF processing demonstration
- Raw data extraction display
- Data cleaning and formatting
- Schema validation results
- Before/after data comparison

### 📊 Analytics
- Key transaction metrics
- Data quality assessment
- Statistical summaries
- Balance analysis
- Performance indicators

### 🧪 Parser Testing
- Custom parser execution
- Output validation
- Performance metrics
- Error handling and feedback

### 📈 Visualizations
- Transaction distribution charts
- Amount distribution histograms
- Balance trend analysis
- Data completeness visualization
- Interactive data explorer

## 🔧 Technology Stack

- **Frontend**: Streamlit (Interactive web interface)
- **Backend**: Python, Pandas, pdfplumber
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas
- **AI Integration**: LangGraph, Groq AI

## 📁 Project Structure

```
ai-agent-challenge/
├── streamlit_app.py              # Main Streamlit application
├── requirements_streamlit.txt     # Streamlit-specific dependencies
├── README_Streamlit.md           # This file
├── data/icici/
│   ├── icici sample.pdf         # Input bank statement
│   └── result.csv               # Expected output format
├── custom_parsers/
│   └── icici_parser.py          # Custom parser implementation
└── agent.py                     # AI agent implementation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Streamlit-specific dependencies
pip install -r requirements_streamlit.txt

# Or install individually
pip install streamlit pandas pdfplumber plotly matplotlib seaborn
```

### 2. Run the Streamlit App

```bash
# Navigate to the project directory
cd ai-agent-challenge

# Run the Streamlit app
streamlit run streamlit_app.py
```

### 3. Access the App

The app will automatically open in your default web browser at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.x.x:8501 (for network access)

## 📱 App Navigation

The app features a sidebar navigation with the following sections:

1. **🏠 Overview** - Project overview, metrics, and data preview
2. **📄 PDF Analysis** - PDF structure analysis and table extraction
3. **🔍 Data Extraction** - Live data extraction and processing
4. **📊 Analytics** - Data analysis and insights
5. **🧪 Parser Testing** - Custom parser validation
6. **📈 Visualizations** - Interactive charts and graphs

## 🎨 Features & Capabilities

### Interactive Data Processing
- Real-time PDF analysis
- Live data extraction demonstration
- Interactive data cleaning
- Schema validation feedback

### Advanced Visualizations
- **Plotly Charts**: Interactive, responsive visualizations
- **Transaction Distribution**: Pie charts showing debit vs credit
- **Amount Distribution**: Histograms for transaction amounts
- **Balance Trends**: Line charts for balance over time
- **Data Quality**: Bar charts for completeness metrics

### Data Analytics
- **Key Metrics**: Transaction counts, amounts, balances
- **Data Quality**: Completeness, missing values, unique counts
- **Statistical Analysis**: Descriptive statistics for all columns
- **Performance Metrics**: Processing time, success rates

### Custom Parser Testing
- **Live Execution**: Test parser in real-time
- **Output Validation**: Compare with expected results
- **Performance Metrics**: Rows processed, data completeness
- **Error Handling**: Graceful error display and feedback

## 🔍 Data Flow

1. **Input**: ICICI bank statement PDF
2. **Processing**: PDF analysis and table extraction
3. **Extraction**: Transaction data extraction
4. **Cleaning**: Data formatting and validation
5. **Analysis**: Statistical analysis and insights
6. **Visualization**: Interactive charts and graphs
7. **Validation**: Schema and parser testing

## 📊 Expected Data Schema

The system processes data with these columns:
- **Date**: Transaction date
- **Description**: Transaction description
- **Debit Amt**: Debit amount (if applicable)
- **Credit Amt**: Credit amount (if applicable)
- **Balance**: Account balance after transaction

## 🎯 Use Cases

### For Developers
- Understand PDF processing workflow
- Test data extraction algorithms
- Validate data processing logic
- Debug parser implementations

### For Analysts
- Explore financial data patterns
- Generate insights from transactions
- Validate data quality
- Create custom visualizations

### For Business Users
- Demonstrate automated processing
- Show data extraction capabilities
- Validate output accuracy
- Understand system capabilities

## 🛠️ Customization

### Adding New Bank Formats
1. Update the `extract_transactions_from_pdf` function
2. Modify column mappings in `clean_and_format_transactions`
3. Update schema validation logic
4. Add new visualization types

### Modifying Visualizations
1. Edit chart creation functions
2. Add new chart types
3. Customize color schemes
4. Modify chart layouts

### Extending Analytics
1. Add new metrics calculations
2. Implement additional statistical tests
3. Create custom data quality measures
4. Add export functionality

## 🐛 Troubleshooting

### Common Issues

**PDF Loading Errors**
- Ensure PDF file exists in `data/icici/` directory
- Check file permissions
- Verify PDF is not corrupted

**Import Errors**
- Install all required dependencies
- Check Python path configuration
- Verify custom_parsers directory structure

**Visualization Issues**
- Ensure data is properly formatted
- Check for empty DataFrames
- Verify column names match expected schema

### Performance Tips

- Use smaller PDFs for testing
- Limit data preview sizes
- Cache expensive computations
- Optimize chart rendering

## 🔮 Future Enhancements

- **File Upload**: Allow users to upload their own PDFs
- **Export Functionality**: Download processed data and charts
- **Batch Processing**: Process multiple files simultaneously
- **Advanced Analytics**: Machine learning insights
- **Real-time Processing**: Live PDF processing
- **Multi-bank Support**: Support for other bank formats

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify file paths and dependencies
3. Review error messages in the app
4. Check the main project README

## 📄 License

This Streamlit app is part of the AI Agent Challenge project. See the main project README for licensing information.

---

**🏦 ICICI Bank Statement Analyzer** | Built with ❤️ using Streamlit & Python 