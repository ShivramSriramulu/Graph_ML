#!/bin/bash

# CORD-19 GraphRAG Streamlit App Launcher
echo "🚀 Starting CORD-19 GraphRAG Streamlit App..."

# Activate virtual environment
source cord_env/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Run Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
