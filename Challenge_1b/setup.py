#!/usr/bin/env python3
"""
Setup and Installation Script for Challenge 1B
Persona-Driven Document Intelligence System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def setup_openai_key():
    """Help user setup OpenAI API key"""
    print("\n🔑 OpenAI API Key Setup")
    print("-" * 30)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print("✅ OpenAI API key found in environment variables")
        print(f"   Key: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("⚠️ OpenAI API key not found in environment variables")
        print("\nTo enable AI-powered section selection, please:")
        print("1. Get an API key from https://platform.openai.com/api-keys")
        print("2. Set it as environment variable:")
        print("   Windows: set OPENAI_API_KEY=your_key_here")
        print("   Linux/Mac: export OPENAI_API_KEY=your_key_here")
        print("\nNote: The system will work without OpenAI key using keyword-based selection")
        return False

def validate_environment():
    """Validate the setup"""
    print("\n🔍 Validating environment...")
    
    try:
        import fitz
        print("✅ PyMuPDF (fitz) imported successfully")
    except ImportError:
        print("❌ PyMuPDF not found. Please install: pip install PyMuPDF")
        return False
    
    try:
        import openai
        print("✅ OpenAI library imported successfully")
    except ImportError:
        print("❌ OpenAI library not found. Please install: pip install openai")
        return False
    
    return True

def create_test_collection():
    """Create a minimal test collection for validation"""
    print("\n🧪 Creating test collection...")
    
    current_dir = Path(__file__).parent
    test_dir = current_dir / "Test_Collection"
    test_dir.mkdir(exist_ok=True)
    
    # Create test input
    test_input = {
        "challenge_info": {
            "challenge_id": "test_case",
            "test_case_name": "validation_test",
            "description": "Test validation"
        },
        "documents": [
            {"filename": "test.pdf", "title": "Test Document"}
        ],
        "persona": {"role": "Test User"},
        "job_to_be_done": {"task": "Validate the system setup"}
    }
    
    with open(test_dir / "challenge1b_input.json", 'w') as f:
        json.dump(test_input, f, indent=2)
    
    # Create minimal PDF outline
    pdfs_dir = test_dir / "PDFs"
    pdfs_dir.mkdir(exist_ok=True)
    
    test_outline = {
        "title": "Test Document",
        "outline": [
            {"level": "H1", "text": "Introduction", "page": 1},
            {"level": "H2", "text": "Getting Started", "page": 2},
            {"level": "H2", "text": "Configuration", "page": 3},
            {"level": "H1", "text": "Conclusion", "page": 4}
        ]
    }
    
    with open(pdfs_dir / "test.json", 'w') as f:
        json.dump(test_outline, f, indent=2)
    
    print(f"✅ Test collection created at {test_dir}")

def main():
    """Main setup function"""
    print("🚀 Challenge 1B Setup Script")
    print("Persona-Driven Document Intelligence System")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return 1
    
    # Step 2: Validate environment
    if not validate_environment():
        print("❌ Setup failed during environment validation")
        return 1
    
    # Step 3: Setup API key
    setup_openai_key()
    
    # Step 4: Create test collection
    try:
        import json
        create_test_collection()
    except Exception as e:
        print(f"❌ Failed to create test collection: {e}")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python test_solution.py")
    print("2. For specific collection: python run_challenge1b.py 'Collection 1'")
    print("3. Check the generated challenge1b_output.json files")
    
    return 0

if __name__ == "__main__":
    exit(main())
