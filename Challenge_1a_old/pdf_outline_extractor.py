class PDFOutlineExtractor:
    def __init__(self):
        pass

    def extract_outline(self, pdf_path):
        doc = fitz.open(pdf_path)
        title = extract_title(doc)
        outline = extract_headings(doc)
        return {"title": title, "outline": outline}
import fitz  # PyMuPDF
import json
import os
import re
def is_bold(span):
    # Heuristic: font name contains 'Bold' or 'Black'
    return 'Bold' in span.get('font', '') or 'Black' in span.get('font', '')

def extract_title(doc):
    # Use the largest, boldest text on the first page as the title, merging consecutive short lines
    import re
    page = doc[0]
    blocks = page.get_text("dict")['blocks']
    candidates = []
    max_size = 0
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if is_bold(span) and span["size"] > max_size and line["bbox"][1] < 200:
                    max_size = span["size"]
    # Collect all bold, largest, top-of-page spans
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if is_bold(span) and abs(span["size"] - max_size) < 1.0 and line["bbox"][1] < 200:
                    # Ignore all-caps short headers
                    if len(text) < 5 and text.isupper():
                        continue
                    if 3 <= len(text.split()) <= 12:
                        candidates.append((line["bbox"], text))
    # Merge consecutive lines if they are close vertically and horizontally
    candidates.sort(key=lambda x: (x[0][1], x[0][0]))
    merged = []
    prev_bbox = None
    for bbox, text in candidates:
        if not merged:
            merged.append(text)
            prev_bbox = bbox
        else:
            # If this line is close to the previous, merge
            if bbox[1] - prev_bbox[3] < 25 and abs(bbox[0] - prev_bbox[0]) < 50:
                merged[-1] += ' ' + text
                prev_bbox = bbox
            else:
                merged.append(text)
                prev_bbox = bbox
    # Remove repeated words/fragments in the merged title
    if merged:
        title = " ".join(merged)
        title = re.sub(r'\b(\w+)( \1\b)+', r'\1', title, flags=re.IGNORECASE)
        return title.strip()
    # Fallback: use largest text near top
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if abs(span["size"] - max_size) < 1.0 and line["bbox"][1] < 200:
                    return span["text"].strip()
    return ""

def extract_headings(doc):
    from collections import Counter
    import re
    outline = []
    font_sizes = Counter()
    left_positions = Counter()
    # First pass: collect all font sizes (rounded) and left positions
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes[round(span["size"])] += 1
                    left_positions[round(line["bbox"][0])] += 1
    # Dynamically select heading sizes: top 2 most frequent, but also consider size gap
    sorted_sizes = sorted(font_sizes.items(), key=lambda x: (-x[1], -x[0]))
    heading_sizes = []
    if sorted_sizes:
        heading_sizes.append(sorted_sizes[0][0])
        # If second most common is at least 80% as frequent and size difference > 1pt, include
        if len(sorted_sizes) > 1 and sorted_sizes[1][1] >= 0.8 * sorted_sizes[0][1] and abs(sorted_sizes[0][0] - sorted_sizes[1][0]) > 1:
            heading_sizes.append(sorted_sizes[1][0])
    # Dynamically extract section keywords from document: words that appear in candidate headings
    candidate_keywords = Counter()
    candidate_texts = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if 5 <= len(text) <= 60 and 2 <= len(text.split()) <= 8:
                        words = [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 3]
                        candidate_keywords.update(words)
                        candidate_texts.append(text)
    # Use top 10 frequent words as section keywords
    section_keywords = set([w for w, _ in candidate_keywords.most_common(10)])
    seen = set()
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_text = span["text"].strip()
                    # Clean up heading text
                    text = raw_text
                    text = re.sub(r'^[^A-Za-z0-9]+', '', text)
                    text = re.sub(r'[:.,;\-]+$', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    if not text or text.lower() in seen:
                        continue
                    size = round(span["size"])
                    is_title_case = text.istitle() or text.isupper()
                    has_keyword = any(kw in text.lower() for kw in section_keywords)
                    is_heading_size = size in heading_sizes
                    is_good_length = 5 <= len(text) <= 60 and 2 <= len(text.split()) <= 8
                    looks_like_sentence = text[0].isupper() and text[1:].islower() and len(text.split()) > 6
                    if (is_heading_size and (is_title_case or has_keyword) and is_good_length and not looks_like_sentence):
                        idx = heading_sizes.index(size)
                        level = f'H{idx+1}'
                        outline.append({"level": level, "text": text, "page": page_num})
                        seen.add(text.lower())
    # Deduplicate headings that are the same except for section numbers
    def strip_section_number(s):
        return re.sub(r"^\d+(\.\d+)*\s*", "", s)
    unique = []
    seen_texts = set()
    for h in outline:
        key = (h["level"], strip_section_number(h["text"]).lower())
        if key not in seen_texts:
            unique.append(h)
            seen_texts.add(key)
    return unique

def process_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    title = extract_title(doc)
    outline = extract_headings(doc)
    result = {"title": title, "outline": outline}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
            try:
                process_pdf(input_path, output_path)
                print(f"Processed {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
import fitz  # PyMuPDF
import json
import os
import re

def is_bold(span):
    # Heuristic: font name contains 'Bold' or 'Black'
    return 'Bold' in span.get('font', '') or 'Black' in span.get('font', '')

def extract_title(doc):
    # Use the largest, boldest text on the first page as the title, merging consecutive short lines
    import re
    page = doc[0]
    blocks = page.get_text("dict")['blocks']
    candidates = []
    max_size = 0
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if is_bold(span) and span["size"] > max_size and line["bbox"][1] < 200:
                    max_size = span["size"]
    # Collect all bold, largest, top-of-page spans
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if is_bold(span) and abs(span["size"] - max_size) < 1.0 and line["bbox"][1] < 200:
                    # Ignore all-caps short headers
                    if len(text) < 5 and text.isupper():
                        continue
                    if 3 <= len(text.split()) <= 12:
                        candidates.append((line["bbox"], text))
    # Merge consecutive lines if they are close vertically and horizontally
    candidates.sort(key=lambda x: (x[0][1], x[0][0]))
    merged = []
    prev_bbox = None
    for bbox, text in candidates:
        if not merged:
            merged.append(text)
            prev_bbox = bbox
        else:
            # If this line is close to the previous, merge
            if bbox[1] - prev_bbox[3] < 25 and abs(bbox[0] - prev_bbox[0]) < 50:
                merged[-1] += ' ' + text
                prev_bbox = bbox
            else:
                merged.append(text)
                prev_bbox = bbox
    # Remove repeated words/fragments in the merged title
    if merged:
        title = " ".join(merged)
        title = re.sub(r'\b(\w+)( \1\b)+', r'\1', title, flags=re.IGNORECASE)
        return title.strip()
    # Fallback: use largest text near top
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if abs(span["size"] - max_size) < 1.0 and line["bbox"][1] < 200:
                    return span["text"].strip()
    return ""

def extract_headings(doc):
    from collections import Counter
    import re
    outline = []
    font_sizes = Counter()
    left_positions = Counter()
    # First pass: collect all font sizes (rounded) and left positions
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes[round(span["size"])] += 1
                    left_positions[round(line["bbox"][0])] += 1
    most_common = [size for size, _ in font_sizes.most_common(4)]
    # If only 3 heading sizes, use up to H3
    if len(most_common) < 4:
        most_common = most_common[:3]
    # Common left margin for headings
    left_margin = left_positions.most_common(1)[0][0] if left_positions else 0
    seen = set()
    prev_heading = None
    prev_props = None
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or text.lower() in seen:
                        continue
                    # Ignore common footer/header patterns
                    if re.match(r"^page \d+$", text.lower()) or re.match(r"^\d+$", text) or text.lower() in {"confidential", "draft"}:
                        continue
                    size = round(span["size"])
                    left = round(line["bbox"][0])
                    if size in most_common:
                        idx = most_common.index(size)
                        level = f'H{idx+1}'
                        outline.append({"level": level, "text": text, "page": page_num})
                        seen.add(text.lower())
                        prev_heading = text
                        prev_props = (page_num, size, left, line["bbox"][3])
    # Deduplicate headings that are the same except for section numbers
    def strip_section_number(s):
        return re.sub(r"^\d+(\.\d+)*\s*", "", s)
    unique = []
    seen_texts = set()
    for h in outline:
        key = (h["level"], strip_section_number(h["text"]).lower())
        if key not in seen_texts:
            unique.append(h)
            seen_texts.add(key)
    return unique
