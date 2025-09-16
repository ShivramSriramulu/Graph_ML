#!/usr/bin/env python3
"""
Deployment script for Streamlit Cloud
"""

import subprocess
import sys
import os
from pathlib import Path

def check_git_status():
    """Check if repository is clean and up to date"""
    print("🔍 Checking Git status...")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Not in a Git repository")
            return False
        
        # Check for uncommitted changes
        if result.stdout.strip():
            print("⚠️ Uncommitted changes detected:")
            print(result.stdout)
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Check if we're on main branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True)
        current_branch = result.stdout.strip()
        
        if current_branch != 'main':
            print(f"⚠️ Not on main branch (currently on {current_branch})")
            response = input("Switch to main branch? (y/N): ")
            if response.lower() == 'y':
                subprocess.run(['git', 'checkout', 'main'])
            else:
                return False
        
        print("✅ Git status OK")
        return True
        
    except Exception as e:
        print(f"❌ Error checking Git status: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("🔍 Checking required files...")
    
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        'README.md',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def push_to_github():
    """Push latest changes to GitHub"""
    print("📤 Pushing to GitHub...")
    
    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True)
        
        # Commit changes
        subprocess.run(['git', 'commit', '-m', 'Update for Streamlit Cloud deployment'], 
                      check=True)
        
        # Push to GitHub
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print("✅ Successfully pushed to GitHub")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error pushing to GitHub: {e}")
        return False

def show_deployment_instructions():
    """Show instructions for Streamlit Cloud deployment"""
    print("\n🚀 Streamlit Cloud Deployment Instructions")
    print("=" * 50)
    
    print("""
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select repository: ShivramSriramulu/Graph_ML
5. Set main file path: streamlit_app.py
6. Add secrets in "Advanced settings":
   
   [openai]
   api_key = "your_openai_api_key_here"
   
7. Click "Deploy!"

Your app will be available at: https://your-app-name.streamlit.app
""")

def main():
    """Main deployment function"""
    print("🧬 CORD-19 GraphRAG Streamlit Cloud Deployment")
    print("=" * 50)
    
    # Check prerequisites
    if not check_git_status():
        print("❌ Git status check failed")
        return
    
    if not check_required_files():
        print("❌ Required files check failed")
        return
    
    # Push to GitHub
    if not push_to_github():
        print("❌ Failed to push to GitHub")
        return
    
    # Show deployment instructions
    show_deployment_instructions()
    
    print("\n✅ Repository is ready for Streamlit Cloud deployment!")
    print("📖 Follow the instructions above to deploy your app.")

if __name__ == "__main__":
    main()
