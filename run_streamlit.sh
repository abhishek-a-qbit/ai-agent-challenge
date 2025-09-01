#!/bin/bash

# ICICI Bank Statement Analyzer - Streamlit App Launcher
# This script launches the Streamlit application

echo "🏦 ICICI Bank Statement Analyzer - Streamlit App"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️  Warning: Streamlit is not installed"
    echo "Installing Streamlit and dependencies..."
    echo ""
    
    # Install Streamlit dependencies
    pip3 install -r requirements_streamlit.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        echo "Please install manually: pip3 install -r requirements_streamlit.txt"
        exit 1
    fi
    
    echo "✅ Dependencies installed successfully!"
    echo ""
fi

# Check if required files exist
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found"
    echo "Please run this script from the ai-agent-challenge directory"
    exit 1
fi

if [ ! -d "data/icici" ]; then
    echo "❌ Error: data/icici directory not found"
    echo "Please ensure the data directory structure is correct"
    exit 1
fi

echo "🚀 Launching Streamlit App..."
echo ""
echo "The app will open in your default web browser"
echo "Local URL: http://localhost:8501"
echo "Network URL: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Launch Streamlit app
streamlit run streamlit_app.py 