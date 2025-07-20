# PDF Outline Extractor for Adobe India Hackathon - Round 1A

This project extracts the document title and all headings (H1, H2, H3) from PDF files, outputting the results in a structured JSON format.

## Features
- Works offline (no cloud APIs)
- Fast (≤ 10 seconds for 50-page PDF)
- Uses PyMuPDF (fitz) for PDF parsing
- Detects headings using font size, boldness, and layout heuristics

## Usage

1. **Install dependencies:**
   ```sh
   pip install pymupdf
   ```

2. **Place your PDF files** in the `input` folder.

3. **Run the script:**
   ```sh
   python pdf_outline_extractor.py
   ```

4. **Results** will be saved as JSON files in the `output` folder, one per PDF.

## Output Format
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "Background", "page": 2 }
  ]
}
```

## Project Structure
- `pdf_outline_extractor.py` — Main script
- `input/` — Place your PDF files here
- `output/` — Extracted outlines will be saved here

## Notes
- Only PDFs with up to 50 pages are supported for optimal speed.
- Heading detection is heuristic-based and may need tuning for some documents.
