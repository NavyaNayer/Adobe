# Copy your collection_template_challenge1b.py script here for each collection
# Run this script in the collection folder to generate challenge1b_output.json

import fitz  # PyMuPDF
import json
import os
from datetime import datetime
import re

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    # --- Minimal 1A logic ---
    def is_bold(span):
        return 'Bold' in span.get('font', '') or 'Black' in span.get('font', '')
    def extract_title(doc):
        page = doc[0]
        blocks = page.get_text("dict")['blocks']
        candidates = []
        max_size = 0
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if is_bold(span) and span["size"] > max_size and line["bbox"][1] < 200:
                        max_size = span["size"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if is_bold(span) and abs(span["size"] - max_size) < 1.0 and line["bbox"][1] < 200:
                        if len(text) < 5 and text.isupper():
                            continue
                        if 3 <= len(text.split()) <= 12:
                            candidates.append((line["bbox"], text))
        candidates.sort(key=lambda x: (x[0][1], x[0][0]))
        merged = []
        prev_bbox = None
        for bbox, text in candidates:
            if not merged:
                merged.append(text)
                prev_bbox = bbox
            else:
                if bbox[1] - prev_bbox[3] < 25 and abs(bbox[0] - prev_bbox[0]) < 50:
                    merged[-1] += ' ' + text
                    prev_bbox = bbox
                else:
                    merged.append(text)
                    prev_bbox = bbox
        if merged:
            title = " ".join(merged)
            title = re.sub(r'\b(\w+)( \1\b)+', r'\1', title, flags=re.IGNORECASE)
            return title.strip()
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if abs(span["size"] - max_size) < 1.0 and line["bbox"][1] < 200:
                        return span["text"].strip()
        return ""
    def extract_headings(doc):
        from collections import Counter
        outline = []
        font_sizes = Counter()
        left_positions = Counter()
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes[round(span["size"])] += 1
                        left_positions[round(line["bbox"][0])] += 1
        most_common = [size for size, _ in font_sizes.most_common(4)]
        if len(most_common) < 4:
            most_common = most_common[:3]
        left_margin = left_positions.most_common(1)[0][0] if left_positions else 0
        seen = set()
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text or text.lower() in seen:
                            continue
                        if re.match(r"^page \d+$", text.lower()) or re.match(r"^\d+$", text) or text.lower() in {"confidential", "draft"}:
                            continue
                        size = round(span["size"])
                        left = round(line["bbox"][0])
                        if size in most_common:
                            idx = most_common.index(size)
                            level = f'H{idx+1}'
                            outline.append({"level": level, "text": text, "page": page_num})
                            seen.add(text.lower())
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
    title = extract_title(doc)
    outline = extract_headings(doc)
    return {"title": title, "outline": outline}

def load_input_json(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    persona = data.get('persona', {})
    job = data.get('job_to_be_done', {})
    documents = data.get('documents', [])
    challenge_info = data.get('challenge_info', {})
    return persona, job, documents, challenge_info

def extract_sections_from_outline(pdf_path, outline):
    # Minimal: Each heading becomes a section with its page number
    sections = []
    for h in outline:
        sections.append({
            'title': h['text'],
            'page': h['page'],
            'content': f"Section: {h['text']} (Page {h['page']})"
        })
    return sections

def rank_and_select_sections(sections, persona, job, top_n=5):
    # Minimal: Score by keyword overlap with persona/job, fallback to order
    keywords = set()
    for v in persona.values():
        keywords.update(str(v).lower().split())
    for v in job.values():
        keywords.update(str(v).lower().split())
    for sec in sections:
        score = sum(1 for word in keywords if word in sec['title'].lower())
        sec['rank'] = score
    sections.sort(key=lambda s: (-s['rank'], s['page']))
    for i, sec in enumerate(sections):
        sec['rank'] = i + 1
    return sections[:top_n]

def extract_refined_subsections(section):
    # Minimal: Just return the section title as refined text
    return f"Refined: {section['title']}"

def generate_output_json(input_docs, persona, job, extracted_sections, subsection_analysis):
    return {
        'input_documents': input_docs,
        'persona': persona,
        'job_to_be_done': job,
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis,
        'generated_at': datetime.now().isoformat()
    }

def main():
    # Assumes current working directory is the collection folder
    input_json = 'challenge1b_input.json'
    pdf_dir = 'PDFs'
    output_json = 'challenge1b_output.json'
    persona, job, documents, challenge_info = load_input_json(input_json)
    extracted_sections = []
    subsection_analysis = []
    input_docs = [doc['filename'] for doc in documents]
    for doc in documents:
        pdf_path = os.path.join(pdf_dir, doc['filename'])
        outline_path = os.path.splitext(pdf_path)[0] + '.json'
        if not os.path.exists(outline_path):
            print(f"Outline not found for {pdf_path}, extracting...")
            outline_data = extract_outline(pdf_path)
            with open(outline_path, 'w', encoding='utf-8') as f:
                json.dump(outline_data, f, ensure_ascii=False, indent=2)
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline_data = json.load(f)
        outline = outline_data['outline']
        sections = extract_sections_from_outline(pdf_path, outline)
        ranked_sections = rank_and_select_sections(sections, persona, job, top_n=5)
        for sec in ranked_sections:
            refined = extract_refined_subsections(sec)
            extracted_sections.append({
                'document': doc['filename'],
                'section_title': sec['title'],
                'importance_rank': sec['rank'],
                'page_number': sec['page']
            })
            subsection_analysis.append({
                'document': doc['filename'],
                'refined_text': refined,
                'page_number': sec['page']
            })
    output = generate_output_json(input_docs, persona, job, extracted_sections, subsection_analysis)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()