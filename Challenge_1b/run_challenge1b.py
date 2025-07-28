#!/usr/bin/env python3
"""
Comprehensive Challenge 1B Solution Runner
Persona-Driven Document Intelligence System

This script:
1. Parses PDFs to extract outlines if needed
2. Runs the persona-driven section selection
3. Generates the final output JSON

Usage:
    python run_challenge1b.py [collection_folder]
"""

import os
import sys
import json
import argparse
from pathlib import Path

def setup_paths():
    """Setup paths for imports"""
    current_dir = Path(__file__).parent
    challenge1a_dir = current_dir.parent / "Challenge_1a"
    
    # Add paths for imports
    if str(challenge1a_dir) not in sys.path:
        sys.path.insert(0, str(challenge1a_dir))
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Check if required dependencies are available"""
    issues = []
    
    # Check Challenge_1a availability
    current_dir = Path(__file__).parent
    challenge1a_dir = current_dir.parent / "Challenge_1a"
    pdf_extractor_file = challenge1a_dir / "pdf_outline_extractor.py"
    
    if not challenge1a_dir.exists():
        issues.append("Challenge_1a directory not found")
    elif not pdf_extractor_file.exists():
        issues.append("pdf_outline_extractor.py not found in Challenge_1a")
    
    # Check selector availability
    selector_file = current_dir / "selector.py"
    if not selector_file.exists():
        issues.append("selector.py not found")
    
    return issues

def parse_pdfs_if_needed(collection_dir: Path):
    """Parse PDFs to generate outline JSON files if they don't exist"""
    try:
        from pdf_outline_extractor import SimplePDFExtractor
    except ImportError as e:
        print(f"❌ Failed to import PDF extractor: {e}")
        print("Make sure Challenge_1a is available and contains pdf_outline_extractor.py")
        return False
    
    pdfs_dir = collection_dir / "PDFs"
    if not pdfs_dir.exists():
        print(f"❌ PDFs directory not found: {pdfs_dir}")
        return False
    
    # Check if there are any PDF files
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print("ℹ️  No PDF files found in PDFs directory")
        return True
    
    extractor = SimplePDFExtractor()
    parsed_count = 0
    
    for pdf_file in pdfs_dir.glob("*.pdf"):
        json_file = pdfs_dir / f"{pdf_file.stem}.json"
        
        # Skip if JSON already exists and is newer
        if json_file.exists() and json_file.stat().st_mtime > pdf_file.stat().st_mtime:
            print(f"✓ Outline already exists: {json_file.name}")
            continue
        
        print(f"📄 Parsing: {pdf_file.name}")
        try:
            result = extractor.extract_outline(str(pdf_file))
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Generated: {json_file.name}")
            parsed_count += 1
            
        except Exception as e:
            print(f"❌ Failed to parse {pdf_file.name}: {e}")
            print(f"   Error details: {str(e)}")
    
    print(f"📊 Parsed {parsed_count} PDF files")
    return True

def run_selector(collection_dir: Path):
    """Run the persona-driven section selector"""
    try:
        from selector import PersonaDrivenSelector
        print("🧠 Running persona-driven section selection...")
        
        # Load input
        input_file = collection_dir / "challenge1b_input.json"
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return False
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"❌ Failed to read input file: {e}")
            return False
        
        # Validate input structure
        required_keys = ['persona', 'job_to_be_done', 'documents']
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            print(f"❌ Input file missing required keys: {missing_keys}")
            return False
        
        # Process
        selector = PersonaDrivenSelector()
        result = selector.process_collection(collection_dir, input_data)
        
        if not result:
            print("❌ Selector returned empty result")
            return False
        
        # Save output
        output_file = collection_dir / "challenge1b_output.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save output file: {e}")
            return False
        
        print("✅ Section selection completed")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import selector: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running selector: {e}")
        return False

def validate_output(collection_dir: Path):
    """Validate the generated output"""
    output_file = collection_dir / "challenge1b_output.json"
    
    if not output_file.exists():
        print("❌ Output file not generated")
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required structure
        if "metadata" not in data or "extracted_sections" not in data:
            print("❌ Output missing required structure")
            return False
        
        sections = data["extracted_sections"]
        if not sections:
            print("❌ No sections extracted")
            return False
        
        print(f"✅ Generated output with {len(sections)} sections")
        
        # Print summary
        for i, section in enumerate(sections, 1):
            title = section.get("section_title", "Unknown")[:50]
            doc = section.get("document", "Unknown")
            page = section.get("page_number", "?")
            print(f"  {i}. {title}... ({doc}, p.{page})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating output: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Challenge 1B solution")
    parser.add_argument("collection", nargs="?", default="Collection 1", 
                       help="Collection folder name (default: Collection 1)")
    parser.add_argument("--parse-only", action="store_true", 
                       help="Only parse PDFs, don't run selector")
    parser.add_argument("--select-only", action="store_true",
                       help="Only run selector, skip PDF parsing")
    
    args = parser.parse_args()
    
    # Setup paths
    setup_paths()
    
    # Check dependencies
    dependency_issues = check_dependencies()
    if dependency_issues:
        print("❌ Dependency issues found:")
        for issue in dependency_issues:
            print(f"   - {issue}")
        return 1
    
    # Determine collection directory
    current_dir = Path(__file__).parent
    collection_dir = current_dir / args.collection
    
    if not collection_dir.exists():
        print(f"❌ Collection directory not found: {collection_dir}")
        return 1
    
    print(f"🎯 Processing collection: {collection_dir.name}")
    
    # Check for input file
    input_file = collection_dir / "challenge1b_input.json"
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    # Step 1: Parse PDFs if needed
    if not args.select_only:
        if not parse_pdfs_if_needed(collection_dir):
            return 1
    
    # Step 2: Run selector
    if not args.parse_only:
        if not run_selector(collection_dir):
            return 1
    
    # Step 3: Validate output
    if not args.parse_only:
        if not validate_output(collection_dir):
            return 1
    
    print("🎉 Challenge 1B completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
