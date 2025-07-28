#!/usr/bin/env python3
"""
Docker Runner for Challenge 1B - Persona-Driven Document Intelligence
Processes collections and generates intelligent section selections
"""

import os
import sys
import json
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '/app')

def main():
    """Main function to process Challenge 1B collections"""
    input_dir = Path('/app/input')
    output_dir = Path('/app/output')
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print("🧠 Challenge 1B - Persona-Driven Document Intelligence")
    print("=" * 60)
    
    # Import required modules
    try:
        from run_challenge1b import setup_paths, parse_pdfs_if_needed, run_selector
        print("✅ Successfully imported Challenge 1B modules")
    except ImportError as e:
        print(f"❌ Failed to import Challenge 1B modules: {e}")
        sys.exit(1)
    
    # Setup paths
    setup_paths()
    
    # Look for collection directories or input files
    collection_dirs = []
    
    # Check for collection-style input (directories with challenge1b_input.json)
    for item in input_dir.iterdir():
        if item.is_dir():
            input_file = item / "challenge1b_input.json"
            if input_file.exists():
                collection_dirs.append(item)
    
    # If no collections found, look for direct input files
    if not collection_dirs:
        # Check for direct challenge1b_input.json in input directory
        direct_input = input_dir / "challenge1b_input.json"
        if direct_input.exists():
            collection_dirs.append(input_dir)
        else:
            print("❌ No valid collection directories or input files found")
            print("Expected: collection directories with challenge1b_input.json files")
            print("Or: challenge1b_input.json directly in input directory")
            sys.exit(1)
    
    print(f"📁 Found {len(collection_dirs)} collection(s) to process")
    
    total_processed = 0
    total_failed = 0
    
    # Process each collection
    for collection_dir in collection_dirs:
        collection_name = collection_dir.name if collection_dir != input_dir else "Main"
        print(f"\n🎯 Processing collection: {collection_name}")
        
        try:
            # Step 1: Parse PDFs if needed (look for PDFs subdirectory)
            pdfs_dir = collection_dir / "PDFs"
            if pdfs_dir.exists():
                print("📄 Parsing PDFs if needed...")
                if not parse_pdfs_if_needed(collection_dir):
                    print(f"⚠️  PDF parsing issues in {collection_name}")
            
            # Step 2: Run persona-driven selector
            print("🧠 Running persona-driven section selection...")
            if run_selector(collection_dir):
                # Copy output to main output directory
                source_output = collection_dir / "challenge1b_output.json"
                if source_output.exists():
                    if collection_dir == input_dir:
                        target_output = output_dir / "challenge1b_output.json"
                    else:
                        target_output = output_dir / f"{collection_name}_output.json"
                    
                    import shutil
                    shutil.copy2(source_output, target_output)
                    print(f"✅ Generated: {target_output.name}")
                    total_processed += 1
                else:
                    print(f"❌ Output file not generated for {collection_name}")
                    total_failed += 1
            else:
                print(f"❌ Failed to process {collection_name}")
                total_failed += 1
                
        except Exception as e:
            print(f"❌ Error processing {collection_name}: {e}")
            total_failed += 1
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {total_processed}")
    print(f"   ❌ Failed: {total_failed}")
    print(f"   📁 Total collections: {len(collection_dirs)}")
    
    if total_failed > 0:
        sys.exit(1)
    
    print("🎉 Challenge 1B completed successfully!")

if __name__ == "__main__":
    main()
