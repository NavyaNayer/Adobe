#!/usr/bin/env python3
"""
Simple test script to demonstrate semantic similarity functionality.
"""

import sys
from pathlib import Path

# Add the path to access the selector
sys.path.append(str(Path(__file__).parent))

def test_semantic_similarity():
    """Test semantic similarity with simple examples."""
    
    # Sample sections from recipes
    sections = [
        {"text": "Vegetable Lasagna", "page": 1, "level": "H3"},
        {"text": "Beef Stroganoff", "page": 2, "level": "H3"},
        {"text": "Coconut Rice", "page": 3, "level": "H3"},
        {"text": "Chicken Parmesan", "page": 4, "level": "H3"},
        {"text": "Potato Salad", "page": 5, "level": "H3"},
        {"text": "Breakfast Burrito", "page": 6, "level": "H3"}
    ]
    
    # Test persona and job
    persona = {"role": "Food Contractor"}
    job = {"task": "Prepare vegetarian buffet-style dinner menu for corporate gathering"}
    
    print("🧪 Testing Semantic Similarity Approach")
    print("=" * 50)
    
    try:
        from selector import PersonaDrivenSelector
        
        # Test with semantic similarity enabled
        selector = PersonaDrivenSelector(use_semantic=True)
        
        if selector.use_semantic:
            print("✅ Semantic similarity model loaded successfully")
            
            # Test the semantic ranking
            results = selector.find_most_relevant_sections(sections, persona, job, top_n=3)
            
            print("\n🎯 Top 3 Most Relevant Sections:")
            for i, (section, score, justification) in enumerate(results, 1):
                print(f"  {i}. {section['text']} (Score: {score:.3f})")
                print(f"     Justification: {justification}")
            
        else:
            print("⚠️ Semantic similarity not available, using keyword matching")
            
    except Exception as e:
        print(f"❌ Error testing semantic similarity: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_semantic_similarity()
    sys.exit(0 if success else 1)
