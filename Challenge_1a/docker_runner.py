#!/usr/bin/env python3
"""
Docker Runner for PDF Outline Extractor
Processes all PDFs in /app/input and generates JSON files in /app/output
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path (pdf_outline_extractor.py is in same directory)
sys.path.insert(0, '/app')

def main():
    """Main function to process all PDFs in input directory"""
    input_dir = Path('/app/input')
    output_dir = Path('/app/output')
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Import the PDF extractor
    try:
        from pdf_outline_extractor import SimplePDFExtractor
        print("✅ Successfully imported SimplePDFExtractor")
    except ImportError as e:
        print(f"❌ Failed to import PDF extractor: {e}")
        print("Available files in current directory:")
        app_dir = Path('/app')
        if app_dir.exists():
            for file in app_dir.iterdir():
                print(f"  - {file.name}")
        sys.exit(1)
    
    # Initialize extractor
    extractor = SimplePDFExtractor()
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("ℹ️  No PDF files found in input directory")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF file(s) to process")
    
    processed_count = 0
    failed_count = 0
    
    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")
        
        try:
            # Extract outline
            result = extractor.extract_outline(str(pdf_file))
            
            # Generate output filename
            output_filename = pdf_file.stem + ".json"
            output_path = output_dir / output_filename
            
            # Save JSON result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Generated: {output_filename}")
            print(f"   Title: {result.get('title', 'No title')}")
            print(f"   Headings: {len(result.get('outline', []))}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {processed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Total files: {len(pdf_files)}")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
