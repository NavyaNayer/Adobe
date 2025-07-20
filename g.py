#!/usr/bin/env python3
"""
PDF Outline Extractor - Enhanced Version
Extracts hierarchical outline (H1, H2, H3) from PDF documents
Outputs structured JSON with title and headings
Enhanced with better title logic and heading detection
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

import fitz  # PyMuPDF

# Fix for "module 'fitz' has no attribute 'open'"
if not hasattr(fitz, "open"):
    try:
        import pymupdf
        fitz = pymupdf
    except ImportError:
        print("PyMuPDF is not installed correctly. Please uninstall any 'fitz' package and install PyMuPDF:")
        print("    pip uninstall fitz")
        print("    pip install PyMuPDF")
        sys.exit(1)


class PDFOutlineExtractor:
    def __init__(self):
        self.max_pages = 50
        self.title_words = set()
        self.font_analysis = {}
        # Known static headings that should be H1
        self.STATIC_H1S = [
            "revision history", "table of contents", "acknowledgements", 
            "references", "introduction", "abstract", "summary",
            "conclusion", "bibliography", "appendix", "glossary"
        ]
        
        # Patterns that indicate body text (not headings)
        self.BODY_TEXT_PATTERNS = [
            r'\b(this document|the following|as described|for example|such as|including but not limited|note that)\b',
            r'\b(provides|describes|contains|includes|shows|demonstrates|explains|outlines|presents)\b',
            r'\b(will be|should be|can be|may be|must be|shall be|would be|could be)\b',
            r'\b(professionals? who|testers? who|individuals? who|people who|those who)\b',
            r'\.\s+[A-Z][a-z]',  # Sentence structure
            r'\b(document|section|chapter|page|figure|table|version|draft)\s+(is|are|will|should|can|may)\b',
            r'www\.',  # URLs
            r'\b\d{4}\b.*\b(version|release|edition|update)\b',  # Version info
        ]

    def is_valid_heading(self, text: str, font_size: float = None) -> bool:
        """Enhanced heading validation with stricter filtering"""
        text = text.strip()
        
        # Skip empty or very short text
        if len(text) <= 2:
            return False
        
        # Valid numbered section patterns (these can be longer but have limits)
        if re.match(r'^\d+\.\s+.{3,80}$', text):  # "1. Introduction" - limited length
            return True
        if re.match(r'^\d+\.\d+\s+.{3,60}$', text):  # "2.1 Intended Audience"
            return True
        if re.match(r'^\d+\.\d+\.\d+\s+.{3,50}$', text):  # "2.1.1 Subsection"
            return True
            
        # Check for static headings (case insensitive) - must be exact or close match
        text_clean = re.sub(r'\s+', ' ', text.lower().strip())
        for static_heading in self.STATIC_H1S:
            # Exact match or starts with the heading
            if text_clean == static_heading or text_clean.startswith(static_heading + " "):
                return len(text) <= 50  # Even static headings shouldn't be too long
        
        # Strict length filtering for non-numbered sections
        if len(text) > 80:  # Much stricter length limit
            return False
            
        # Enhanced body text detection
        text_lower = text.lower()
        for pattern in self.BODY_TEXT_PATTERNS:
            if re.search(pattern, text_lower):
                return False
        
        # Check for multiple complete sentences (strong indicator of body text)
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences 
                              if len(s.strip()) > 8 and not re.match(r'^\d+(\.\d+)*$', s.strip())]
        if len(meaningful_sentences) > 1:
            return False
            
        # Reject text with too many common words (likely body text)
        words = text_lower.split()
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                       'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'will', 'should', 
                       'can', 'may', 'must', 'shall', 'would', 'could', 'have', 'has', 'had']
        
        if len(words) > 4:  # Only check longer texts
            common_word_ratio = sum(1 for word in words if word in common_words) / len(words)
            if common_word_ratio > 0.4:  # Too many common words
                return False
            
        return True

    def get_heading_level(self, text: str) -> Optional[str]:
        """Enhanced heading level detection with improved logic"""
        text = text.strip()
        
        # Numbered sections - more precise matching
        if re.match(r'^\d+\.\d+\.\d+\s+', text):  # "2.1.1 Something"
            return "H3"
        elif re.match(r'^\d+\.\d+\s+', text):     # "2.1 Something"
            return "H2"
        elif re.match(r'^\d+\.\s+', text):        # "1. Introduction"
            return "H1"
            
        # Static headings are always H1 - but be more restrictive
        text_clean = re.sub(r'\s+', ' ', text.lower().strip())
        for static_heading in self.STATIC_H1S:
            if (text_clean == static_heading or 
                (text_clean.startswith(static_heading + " ") and len(text) <= 50)):
                return "H1"
        
        return None

    def analyze_fonts(self, pages_data: List[Dict]) -> Dict[str, Any]:
        """Analyze font usage patterns across the document"""
        font_sizes = Counter()
        font_styles = defaultdict(list)
        
        for page in pages_data:
            for span in page["spans"]:
                size = round(span["font_size"], 1)
                font_sizes[size] += 1
                
                is_bold = bool(span.get("flags", 0) & 2**4)
                font_styles[size].append({
                    'bold': is_bold,
                    'text': span["text"],
                    'page': page["page"]
                })
        
        common_sizes = font_sizes.most_common()
        
        # Determine body text size (most common size)
        body_text_size = common_sizes[0][0] if common_sizes else 12
        
        # Find potential heading sizes (significantly larger than body text)
        heading_sizes = []
        for size, count in common_sizes:
            # Must be larger than body text and not too common (not body text)
            if (size > body_text_size and 
                count < len(pages_data) * 5 and  # Not too frequent
                size >= body_text_size + 1):     # Meaningful size difference
                heading_sizes.append(size)
        
        heading_sizes.sort(reverse=True)
        
        return {
            'font_sizes': font_sizes,
            'body_text_size': body_text_size,
            'heading_sizes': heading_sizes[:3],  # Top 3 potential heading sizes
            'font_styles': font_styles
        }

    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            if len(doc) > self.max_pages:
                raise ValueError(f"PDF has {len(doc)} pages. Maximum allowed is {self.max_pages}.")

            pages_data = []
            print(f"Processing {len(doc)} pages...")

            # Collect all spans from all pages
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                page_spans = []
                
                for b in blocks:
                    if "lines" in b:
                        for line in b["lines"]:
                            line_text = ""
                            line_font_size = 0
                            line_flags = 0
                            
                            # Reconstruct full line from spans
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    line_font_size = max(line_font_size, span["size"])
                                    line_flags |= span.get("flags", 0)
                            
                            line_text = line_text.strip()
                            if not line_text:
                                continue
                                
                            span_data = {
                                "text": line_text,
                                "font_size": round(line_font_size, 1),
                                "flags": line_flags,
                                "page": page_num + 1,
                            }
                            page_spans.append(span_data)
                
                pages_data.append({"page": page_num + 1, "spans": page_spans})

            # Analyze font patterns
            self.font_analysis = self.analyze_fonts(pages_data)
            
            # Extract title and outline
            title = self.extract_title(pages_data)
            self.title_words = set(word.lower() for word in title.split())
            
            outline = self.extract_outline_structure(pages_data)
            outline = self.clean_and_validate_outline(outline)

            return {
                "title": title,
                "outline": outline
            }

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def extract_title(self, pages_data: List[Dict]) -> str:
        """Extract document title from first page's largest font spans"""
        if not pages_data or not pages_data[0]["spans"]:
            return "Untitled Document"
        spans = pages_data[0]["spans"]
        # Get first two significant text blocks
        candidates = []
        for span in sorted(spans, key=lambda x: -x["font_size"]):
            text = span["text"].strip()
            if len(text) > 3 and not any(kw in text.lower() for kw in ["contents", "copyright"]):
                candidates.append(text)
                if len(candidates) == 2:
                    break
        return "  ".join(candidates) + "  " if candidates else "Untitled Document"

    def identify_heading(self, span: Dict) -> Optional[Dict[str, str]]:
        """Enhanced heading identification with stricter validation"""
        text = span["text"].strip()
        font_size = span["font_size"]
        flags = span.get("flags", 0)
        
        # First check if it's a valid heading format
        if not self.is_valid_heading(text, font_size):
            return None
            
        # Get level from text pattern (numbered sections and static headings)
        level = self.get_heading_level(text)
        if level:
            return {"level": level, "text": text}
            
        # Font-based heading detection for non-numbered headings
        body_size = self.font_analysis.get('body_text_size', 12)
        is_bold = bool(flags & 2**4)
        heading_sizes = self.font_analysis.get('heading_sizes', [])
        
        # Must be significantly larger than body text
        if font_size >= body_size + 1.5:
            # Determine level based on font size relative to other headings
            if heading_sizes:
                if len(heading_sizes) >= 1 and font_size >= heading_sizes[0] - 0.5:
                    level = "H1"
                elif len(heading_sizes) >= 2 and font_size >= heading_sizes[1] - 0.5:
                    level = "H2"
                else:
                    level = "H3"
            else:
                # Fallback if no clear heading sizes detected
                if font_size >= body_size + 3:
                    level = "H1"
                elif font_size >= body_size + 2:
                    level = "H2"
                else:
                    level = "H3"
                    
            # Additional validation for font-based headings
            # Must be reasonably short and either bold or significantly larger
            if (len(text) <= 60 and 
                (is_bold or font_size >= body_size + 2) and
                not self._looks_like_body_text(text)):
                return {"level": level, "text": text}
                
        return None
    
    def _looks_like_body_text(self, text: str) -> bool:
        """Additional check to see if text looks like body text"""
        text_lower = text.lower()
        
        # Strong indicators of body text
        body_indicators = [
            r'\b(this|that|these|those|the following|as mentioned|as described|for example)\b',
            r'\b(professionals? who|individuals? who|testers? who|people who)\b',
            r'\b(will be|should be|can be|may be|must be|shall be)\b',
            r'\b(document|section|chapter|provides|describes|contains|includes)\b',
            r'[.!?]\s+[A-Z]',  # Multiple sentences
        ]
        
        for pattern in body_indicators:
            if re.search(pattern, text_lower):
                return True
                
        return False

    def extract_outline_structure(self, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract outline with improved handling: if 'table of contents' heading exists on a page,
        only extract that heading from that page."""
        outline = []
        seen_headings = {}  # Track headings with their pages and levels

        for page in pages_data:
            # First, scan to check if a "Table of Contents" heading exists on this page
            toc_heading = None
            for span in page["spans"]:
                heading_info = self.identify_heading(span)
                if heading_info:
                    text = heading_info["text"].strip().lower()
                    if text.startswith("table of contents"):
                        toc_heading = heading_info
                        break

            if toc_heading:
                # Extract only the "Table of Contents" heading from this page
                text = toc_heading["text"].strip()
                level = toc_heading["level"]
                page_num = page["page"]

                # Skip if title duplicate
                if self._is_title_duplicate(text):
                    continue

                text_normalized = self._normalize_heading_text(text)
                if text_normalized not in seen_headings:
                    seen_headings[text_normalized] = {'level': level, 'page': page_num}
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": page_num
                    })
                # Skip all other headings on this page
                continue

            # If no TOC heading, fall back to normal heading extraction
            for span in page["spans"]:
                heading_info = self.identify_heading(span)
                if heading_info:
                    text = heading_info["text"].strip()
                    level = heading_info["level"]
                    page_num = page["page"]

                    if self._is_title_duplicate(text):
                        continue

                    text_normalized = self._normalize_heading_text(text)

                    if text_normalized in seen_headings:
                        existing_entry = seen_headings[text_normalized]

                        level_priority = {'H1': 1, 'H2': 2, 'H3': 3}

                        should_replace = (
                            (level == existing_entry['level'] and page_num > existing_entry['page'] + 2) or
                            level_priority.get(level, 4) < level_priority.get(existing_entry['level'], 4)
                        )

                        if should_replace:
                            outline = [item for item in outline if not (
                                self._normalize_heading_text(item['text']) == text_normalized and
                                item['page'] == existing_entry['page']
                            )]
                            seen_headings[text_normalized] = {'level': level, 'page': page_num}
                        else:
                            continue
                    else:
                        seen_headings[text_normalized] = {'level': level, 'page': page_num}

                    outline.append({
                        "level": level,
                        "text": text,
                        "page": page_num
                    })

        return outline

    def _normalize_heading_text(self, text: str) -> str:
        """Normalize heading text for better duplicate detection"""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove trailing punctuation and whitespace
        normalized = re.sub(r'[.,:;!?\s]+$', '', normalized)
        
        # Remove leading/trailing quotes
        normalized = normalized.strip('"\'')
        
        return normalized

    def _is_title_duplicate(self, text: str) -> bool:
        """Check if heading text is a duplicate of the document title"""
        if not self.title_words:
            return False
            
        text_words = set(word.lower() for word in text.split())
        
        # Remove common words for comparison
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        text_words -= common_words
        title_words_clean = self.title_words - common_words
        
        # Check for significant overlap
        if len(title_words_clean) > 0:
            overlap = len(text_words & title_words_clean) / len(title_words_clean)
            return overlap > 0.6  # Slightly more permissive
            
        return False

    def clean_and_validate_outline(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate the extracted outline"""
        if not outline:
            return []
            
        # Sort by page number first
        cleaned = sorted(outline, key=lambda x: x["page"])
        
        validated = []
        prev_text_normalized = set()
        
        for item in cleaned:
            text = item["text"].strip()
            level = item["level"]
            page_num = item["page"]
            
            # Normalize for final duplicate check
            text_normalized = self._normalize_heading_text(text)
            
            # Skip if we've already seen this exact normalized text
            if text_normalized in prev_text_normalized:
                continue
            
            # Final validation
            if (self.is_valid_heading(text) and 
                3 <= len(text) <= 100 and  # Reasonable length bounds
                not self._looks_like_body_text(text)):
                
                # Clean up text formatting
                text = re.sub(r'\s+', ' ', text).strip()
                # Remove trailing punctuation except for numbered sections
                if not re.match(r'^\d+(\.\d+)*\s+', text):
                    text = re.sub(r'[.,:;!?]+$', '', text)
                
                validated.append({
                    "level": level,
                    "text": text,
                    "page": page_num
                })
                
                prev_text_normalized.add(text_normalized)
        
        return validated


def main():
    parser = argparse.ArgumentParser(description='Extract hierarchical outline from PDF documents')
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: outline.json)')
    parser.add_argument('--preview', action='store_true', help='Show preview of extracted outline')

    args = parser.parse_args()

    # Validate input file
    pdf_path = Path(args.input_pdf)
    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)

    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: File '{pdf_path}' is not a PDF.")
        sys.exit(1)

    # Set output path
    output_path = args.output if args.output else 'outline.json'

    try:
        # Extract outline
        print(f"Extracting outline from: {pdf_path}")
        extractor = PDFOutlineExtractor()
        result = extractor.extract_outline(str(pdf_path))

        # Ensure proper data types
        result["title"] = str(result["title"]).strip()
        result["outline"] = [
            {
                "level": str(item["level"]),
                "text": str(item["text"]).strip(),
                "page": int(item["page"])
            }
            for item in result["outline"]
        ]

        # Save JSON output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write('\n')

        # Show preview if requested
        if args.preview:
            print("\n" + "="*60)
            print("EXTRACTED OUTLINE PREVIEW")
            print("="*60)
            print(f"Title: {result['title']}")
            print(f"Total headings: {len(result['outline'])}")

            h1_count = sum(1 for item in result['outline'] if item['level'] == 'H1')
            h2_count = sum(1 for item in result['outline'] if item['level'] == 'H2')
            h3_count = sum(1 for item in result['outline'] if item['level'] == 'H3')

            print(f"H1: {h1_count}, H2: {h2_count}, H3: {h3_count}")
            print("\nOutline Structure:")
            print("-" * 40)

            for item in result['outline']:
                indent = "  " * (int(item['level'][1]) - 1)
                print(f"{indent}{item['level']}: {item['text']} (Page {item['page']})")

        print(f"\n✅ Outline extracted successfully!")
        print(f"📄 Title: {result['title']}")
        print(f"📋 Total headings: {len(result['outline'])}")
        print(f"💾 Output saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()