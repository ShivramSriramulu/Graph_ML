#!/usr/bin/env python3
"""
CORD-19 GraphRAG Streamlit App Launcher with Status Check
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import os

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if virtual environment exists
    if not Path("cord_env").exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return False
    
    # Check if data files exist
    required_files = [
        "results/cord19_graph.gml",
        "results/embeddings.npy", 
        "results/faiss_index.bin",
        "2020-04-10/metadata.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run the pipeline first: python run_pipeline.py")
        return False
    
    print("✅ All requirements met!")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import streamlit
        import plotly
        import pyvis
        import pandas
        import networkx
        import numpy
        import faiss
        import torch
        import transformers
        import spacy
        import openai
        print("✅ All dependencies installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements_streamlit.txt")
        return False

def launch_streamlit():
    """Launch the Streamlit app"""
    print("🚀 Launching CORD-19 GraphRAG Streamlit App...")
    
    # Set environment variables
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🌐 Starting server...")
        print("📱 The app will open in your browser automatically")
        print("🔗 Or visit: http://localhost:8501")
        print("\n" + "="*50)
        print("Press Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    """Main launcher function"""
    print("🧬 CORD-19 GraphRAG Streamlit App Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Launch app
    launch_streamlit()

if __name__ == "__main__":
    main()
