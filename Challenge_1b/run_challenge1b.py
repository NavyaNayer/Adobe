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
    common_dir = current_dir / "common.py"
    
    # Add paths for imports
    if str(challenge1a_dir) not in sys.path:
        sys.path.insert(0, str(challenge1a_dir))
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))

def parse_pdfs_if_needed(collection_dir: Path):
    """Parse PDFs to generate outline JSON files if they don't exist"""
    from pdf_outline_extractor import PDFOutlineExtractor
    
    pdfs_dir = collection_dir / "PDFs"
    if not pdfs_dir.exists():
        print(f"❌ PDFs directory not found: {pdfs_dir}")
        return False
    
    extractor = PDFOutlineExtractor()
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
        
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Process
        selector = PersonaDrivenSelector()
        result = selector.process_collection(collection_dir, input_data)
        
        # Save output
        output_file = collection_dir / "challenge1b_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("✅ Section selection completed")
        return True
        
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
