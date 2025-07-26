#!/usr/bin/env python3
"""
Test and Validation Script for Challenge 1B Solution
Tests the persona-driven document intelligence system
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

def setup_environment():
    """Setup the test environment"""
    current_dir = Path(__file__).parent
    
    # Add paths for imports
    challenge1a_dir = current_dir.parent / "Challenge_1a"
    common_dir = current_dir / "common.py"
    
    if str(challenge1a_dir) not in sys.path:
        sys.path.insert(0, str(challenge1a_dir))
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))

def test_collection(collection_name: str) -> Dict:
    """Test a specific collection"""
    current_dir = Path(__file__).parent
    collection_dir = current_dir / collection_name
    
    print(f"\n🧪 Testing {collection_name}")
    print("=" * 50)
    
    results = {
        "collection": collection_name,
        "success": False,
        "input_exists": False,
        "pdfs_found": 0,
        "sections_extracted": 0,
        "processing_time": 0,
        "errors": []
    }
    
    try:
        # Check if collection exists
        if not collection_dir.exists():
            results["errors"].append(f"Collection directory not found: {collection_dir}")
            return results
        
        # Check input file
        input_file = collection_dir / "challenge1b_input.json"
        if not input_file.exists():
            results["errors"].append("Input file (challenge1b_input.json) not found")
            return results
        
        results["input_exists"] = True
        
        # Load and validate input
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        print(f"📋 Persona: {input_data.get('persona', {}).get('role', 'Unknown')}")
        print(f"🎯 Task: {input_data.get('job_to_be_done', {}).get('task', 'Unknown')}")
        
        # Check PDFs
        pdfs_dir = collection_dir / "PDFs"
        if pdfs_dir.exists():
            pdf_files = list(pdfs_dir.glob("*.pdf"))
            results["pdfs_found"] = len(pdf_files)
            print(f"📄 Found {len(pdf_files)} PDF files")
            
            for pdf in pdf_files:
                print(f"  - {pdf.name}")
        
        # Test the selector
        start_time = time.time()
        
        # Change to collection directory
        original_cwd = os.getcwd()
        os.chdir(collection_dir)
        
        try:
            # Import and run selector
            from selector import main as selector_main
            selector_main()
            
            # Check output
            output_file = collection_dir / "challenge1b_output.json"
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                
                sections = output_data.get("extracted_sections", [])
                results["sections_extracted"] = len(sections)
                
                print(f"✅ Generated {len(sections)} sections")
                
                # Validate output structure
                if validate_output_structure(output_data):
                    results["success"] = True
                    print("✅ Output structure is valid")
                else:
                    results["errors"].append("Invalid output structure")
                
            else:
                results["errors"].append("Output file not generated")
        
        finally:
            os.chdir(original_cwd)
        
        results["processing_time"] = time.time() - start_time
        
    except Exception as e:
        results["errors"].append(f"Execution error: {str(e)}")
    
    return results

def validate_output_structure(output_data: Dict) -> bool:
    """Validate the structure of output data"""
    required_fields = ["metadata", "extracted_sections"]
    
    for field in required_fields:
        if field not in output_data:
            print(f"❌ Missing required field: {field}")
            return False
    
    metadata = output_data["metadata"]
    required_metadata = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
    
    for field in required_metadata:
        if field not in metadata:
            print(f"❌ Missing metadata field: {field}")
            return False
    
    sections = output_data["extracted_sections"]
    if not isinstance(sections, list):
        print("❌ extracted_sections must be a list")
        return False
    
    for i, section in enumerate(sections):
        required_section_fields = ["document", "section_title", "page_number"]
        for field in required_section_fields:
            if field not in section:
                print(f"❌ Section {i+1} missing field: {field}")
                return False
    
    return True

def generate_report(results: List[Dict]):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("🏆 CHALLENGE 1B TEST REPORT")
    print("="*60)
    
    total_collections = len(results)
    successful_collections = sum(1 for r in results if r["success"])
    
    print(f"Collections Tested: {total_collections}")
    print(f"Successful: {successful_collections}")
    print(f"Failed: {total_collections - successful_collections}")
    print(f"Success Rate: {successful_collections/total_collections*100:.1f}%")
    
    print("\n📊 DETAILED RESULTS:")
    print("-" * 60)
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['collection']}")
        print(f"   PDFs: {result['pdfs_found']}, Sections: {result['sections_extracted']}")
        print(f"   Time: {result['processing_time']:.2f}s")
        
        if result["errors"]:
            print(f"   Errors: {', '.join(result['errors'])}")
        print()
    
    # Overall assessment
    if successful_collections == total_collections:
        print("🎉 ALL TESTS PASSED! Solution is ready for submission.")
    elif successful_collections > 0:
        print("⚠️ PARTIAL SUCCESS. Some collections failed - review errors above.")
    else:
        print("❌ ALL TESTS FAILED. Solution needs major fixes.")

def main():
    """Main test runner"""
    print("🚀 Challenge 1B Solution Tester")
    print("Testing Persona-Driven Document Intelligence System")
    
    setup_environment()
    
    # Find all collection directories
    current_dir = Path(__file__).parent
    collections = [d.name for d in current_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("Collection")]
    
    if not collections:
        print("❌ No collection directories found!")
        return 1
    
    print(f"📁 Found {len(collections)} collection(s): {', '.join(collections)}")
    
    # Test each collection
    results = []
    for collection in sorted(collections):
        result = test_collection(collection)
        results.append(result)
    
    # Generate report
    generate_report(results)
    
    # Return appropriate exit code
    successful = sum(1 for r in results if r["success"])
    return 0 if successful == len(results) else 1

if __name__ == "__main__":
    exit(main())
