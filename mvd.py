#!/usr/bin/env python3
"""
PDF Outline Extractor - Hackathon Version
Extracts hierarchical outline (H1, H2, H3) from PDF documents
Outputs structured JSON with title and headings
Optimized for Adobe India Hackathon 2025 (Round 1A)
"""

import os
import sys
import json
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_PAGES = 50
STATIC_H1S = [
    "revision history", "table of contents", "acknowledgements",
    "references", "introduction", "abstract", "summary",
    "conclusion", "bibliography", "appendix", "glossary"
]

def is_valid_heading(text: str) -> bool:
    text = text.strip()
    # Filter out very short, incomplete, or non-heading lines
    if not text or len(text) > 120 or len(text) <= 5:
        return False
    if text.lower() in STATIC_H1S:
        return True
    if re.match(r'^(chapter|section|appendix)\b', text, re.I):
        return True
    if text[0].isdigit() and '.' in text and len(text.split()) > 2:
        return True
    # Avoid lines that are just a few words or fragments
    if len(text.split()) < 3:
        return False
    # Heuristic: must have at least 1 uppercase in first 8 chars and not end with punctuation
    if any(c.isupper() for c in text[:8]) and not text.endswith(('.', ':', ';')):
        return True
    if text.isupper() and 3 <= len(text.split()) <= 10:
        return True
    return False

def get_heading_level(text: str, font_size: float, heading_sizes: List[float]) -> Optional[str]:
    """Assign H1, H2, H3 based on font size, patterns, and keywords"""
    if re.match(r'^(appendix|chapter)\b', text, re.I):
        return "H1"
    if re.match(r'^\d+\.\d+\.\d+\s+', text):
        return "H3"
    elif re.match(r'^\d+\.\d+\s+', text):
        return "H2"
    elif re.match(r'^\d+\.\s+', text):
        return "H1"
    if heading_sizes:
        if font_size >= heading_sizes[0] - 0.5:
            return "H1"
        elif len(heading_sizes) > 1 and font_size >= heading_sizes[1] - 0.5:
            return "H2"
        else:
            return "H3"
    return None

def analyze_fonts(doc: fitz.Document) -> Dict[str, Any]:
    font_sizes = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        size = round(span["size"], 1)
                        font_sizes[size] += 1
    most_common_sizes = font_sizes.most_common()
    body_size = most_common_sizes[0][0] if most_common_sizes else 12
    heading_sizes = [size for size, _ in most_common_sizes if size > body_size + 1]
    return {
        "body_size": body_size,
        "heading_sizes": heading_sizes[:3]  # Top 3
    }

def extract_title(doc: fitz.Document, heading_sizes: List[float]) -> str:
    """Get document title as largest text block on first page, fallback to metadata"""
    page = doc[0]
    largest_text = ""
    max_size = 0
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"]).strip()
                size = max(span["size"] for span in line["spans"])
                if size > max_size and 2 <= len(text.split()) <= 15 and text.isprintable():
                    max_size = size
                    largest_text = text
    if largest_text:
        return largest_text
    meta_title = doc.metadata.get("title")
    if meta_title and meta_title.strip():
        return meta_title.strip()
    return "Untitled Document"

def extract_headings(doc: fitz.Document, heading_sizes: List[float]) -> List[Dict[str, Any]]:
    """Extract hierarchical headings across pages, merge fragments, avoid duplicates, and filter noise"""
    outline = []
    seen = set()
    for page_num, page in enumerate(doc, start=1):
        if page_num > MAX_PAGES:
            break
        prev_text = None
        prev_level = None
        prev_size = None
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    text = " ".join(span["text"] for span in line["spans"]).strip()
                    size = max(span["size"] for span in line["spans"])
                    # Merge consecutive lines/fragments into one if likely part of the same heading
                    if prev_text:
                        # If both lines are short or don't end with punctuation, merge
                        if (len(prev_text.split()) < 10 or not prev_text.endswith(('.', ':', ';'))) and (len(text.split()) < 10 or not text.endswith(('.', ':', ';'))):
                            merged = prev_text + " " + text
                            if is_valid_heading(merged):
                                level = get_heading_level(merged, max(size, prev_size), heading_sizes)
                                key = (merged.lower(), level)
                                if level and key not in seen:
                                    outline[-1]["text"] = merged
                                    outline[-1]["level"] = level
                                    seen.add(key)
                                prev_text = None
                                continue
                    if is_valid_heading(text):
                        # Filter out headings that are likely fragments (e.g. very short, or only 1-2 words)
                        if len(text.split()) < 4 and not text.endswith(('.', ':', ';')):
                            prev_text = text
                            prev_level = None
                            prev_size = size
                            continue
                        level = get_heading_level(text, size, heading_sizes)
                        key = (text.lower(), level)
                        if level and key not in seen:
                            outline.append({
                                "level": level,
                                "text": text,
                                "page": page_num
                            })
                            seen.add(key)
                        prev_text = text
                        prev_level = level
                        prev_size = size
                    else:
                        prev_text = text
                        prev_level = None
                        prev_size = size
    return outline

def generate_json(title: str, outline: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "title": title,
        "outline": outline
    }

def process_pdf(input_path: Path, output_path: Path):
    try:
        if not input_path.exists() or not input_path.is_file():
            logging.error(f"File not found: {input_path}")
            return
        logging.info(f"Processing {input_path}")
        doc = fitz.open(str(input_path))
        fonts = analyze_fonts(doc)
        title = extract_title(doc, fonts["heading_sizes"])
        outline = extract_headings(doc, fonts["heading_sizes"])
        result = generate_json(title, outline)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved outline to {output_path}")
    except Exception as e:
        logging.error(f"Failed to process {input_path}: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF Outline Extractor GOD LEVEL")
    parser.add_argument("input_pdf", nargs="?", help="Path to input PDF file (single mode)")
    parser.add_argument("-o", "--output", help="Output JSON file path (single mode)")
    parser.add_argument("--input-dir", help="Input directory for batch mode (default: ./input)")
    parser.add_argument("--output-dir", help="Output directory for batch mode (default: ./output)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Max pages to scan per PDF")
    args = parser.parse_args()

    global MAX_PAGES
    MAX_PAGES = args.max_pages

    if args.input_pdf:
        input_path = Path(args.input_pdf)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".json")
        process_pdf(input_path, output_path)
    else:
        # Batch mode: process all PDFs in input_dir
        input_dir = Path(args.input_dir) if args.input_dir else Path("./input")
        output_dir = Path(args.output_dir) if args.output_dir else Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logging.warning(f"No PDF files found in {input_dir}")
            return

        for pdf_file in pdf_files:
            output_file = output_dir / (pdf_file.stem + ".json")
            process_pdf(pdf_file, output_file)


if __name__ == "__main__":
    main()
