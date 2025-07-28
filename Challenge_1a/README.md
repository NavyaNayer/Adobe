# PDF Outline Extractor - Challenge 1A

## Overview
This solution extracts structured outlines (titles and headings) from PDF documents, supporting multilingual and non-Latin scripts. It is designed for robust, offline, and automatic processing in a Dockerized environment.

## Solution Logic

### 1. PDF Parsing
- Uses **PyMuPDF** to parse PDF files and extract text, font sizes, and layout information from each page.
- Processes all PDFs in the `/app/input` directory and writes JSON outputs to `/app/output`.

### 2. Title Detection
- The first page is scanned for the largest font size in the top 40% of the page.
- All lines with the largest font size in this region are considered as possible titles.
- Titles are cleaned and deduplicated using whitespace normalization.

### 3. Heading Extraction
- For each page, all text blocks are analyzed.
- Font sizes are collected and sorted to determine heading levels (H1, H2, H3, H4).
- Section numbers (e.g., 1.2, 2.3.4) are detected using regex for additional heading structure.
- For non-Latin scripts (e.g., Japanese), special handling is used:
  - Lines with only punctuation or numbers are ignored.
- Headings are deduplicated and sorted by page and section order.

### 4. Multi-language Support
- **langdetect** is used to detect the document language.
- If the document is non-English but uses a Latin script, a multilingual extractor is used.
- For non-Latin scripts, a dedicated extractor handles script-specific quirks (e.g., Japanese TOC formatting).

### 5. Output
- For each PDF, a JSON file is generated with the structure:
  ```json
  {
    "title": "Document Title",
    "outline": [
      { "level": "H1", "text": "Section Title", "page": 0 },
      ...
    ]
  }
  ```
- The output is designed for easy downstream use and evaluation.

## Key Features
- **Robust title and heading detection** using font size, position, and section numbering.
- **Dot leader and page number cleanup** for TOC-style headings, especially in Japanese/Chinese PDFs.
- **Automatic language detection** and script-specific extraction logic.
- **Offline, reproducible, and Dockerized** for consistent results.

## Usage
See `DOCKER_INSTRUCTIONS.md` for build and run commands. Place your PDFs in the `input/` directory and collect results from `output/`.

## Dependencies
- PyMuPDF==1.26.3
- langdetect==1.0.9

## Author
Team Smart Builders
Diksha Khandelwal
Manvendra Singh Tanwar
Navya Nayer
