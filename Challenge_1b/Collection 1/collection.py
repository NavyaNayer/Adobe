# Copy your collection_template_challenge1b.py script here for each collection
# Run this script in the collection folder to generate challenge1b_output.json


import fitz  # PyMuPDF
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Challenge_1a')))
from pdf_outline_extractor import PDFOutlineExtractor
import json
from datetime import datetime
import re

def extract_outline(pdf_path):
    # Use the enhanced extractor from Challenge_1a
    extractor = PDFOutlineExtractor()
    return extractor.extract_outline(pdf_path)

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
    # Extract and summarize actual paragraph text from the relevant PDF page and section
    pdf_path = section.get('pdf_path')
    title = section.get('section_title', '')
    page_number = section.get('page_number')
    if not pdf_path or not title or not page_number:
        return f"Refined: {title}"
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        text = page.get_text("text")
        # Try to find the section title in the page text and extract the following paragraph
        pattern = re.escape(title)
        match = re.search(pattern + r"[\s\n]*([^.\n]{20,}\.)", text, re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            return f"Refined: {title}: {summary}"
        # Fallback: extract first 2 sentences after the title
        idx = text.lower().find(title.lower())
        if idx != -1:
            after = text[idx + len(title):]
            sentences = re.split(r'(?<=[.!?])\s+', after)
            summary = ' '.join(sentences[:2]).strip()
            if summary:
                return f"Refined: {title}: {summary}"
        # Fallback: first 2 sentences from page
        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary = ' '.join(sentences[:2]).strip()
        return f"Refined: {title}: {summary}"
    except Exception as e:
        return f"Refined: {title}"

def generate_output_json(input_docs, persona, job, extracted_sections, subsection_analysis):
    # Format output as per the requested structure
    return {
        "metadata": {
            "input_documents": input_docs,
            "persona": persona.get("role", ""),
            "job_to_be_done": job.get("task", ""),
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def main():
    # Assumes current working directory is the collection folder
    input_json = 'challenge1b_input.json'
    pdf_dir = 'PDFs'
    output_json = 'challenge1b_output.json'
    persona, job, documents, challenge_info = load_input_json(input_json)
    all_sections = []
    input_docs = [doc['filename'] for doc in documents]
    section_objs_by_doc_page = {}
    travel_keywords = {"guide", "adventures", "experiences", "tips", "nightlife", "packing", "culinary", "entertainment", "coastal", "comprehensive", "general", "cities", "culture", "restaurants", "hotels", "things", "do", "wine", "festivals", "summary"}
    for doc in documents:
        pdf_path = os.path.join(pdf_dir, doc['filename'])
        outline_path = os.path.splitext(pdf_path)[0] + '.json'
        print(f"Extracting outline for {pdf_path}...")
        outline_data = extract_outline(pdf_path)
        with open(outline_path, 'w', encoding='utf-8') as f:
            json.dump(outline_data, f, ensure_ascii=False, indent=2)
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline_data = json.load(f)
        outline = outline_data['outline']
        print(f"\nExtracted outline for {doc['filename']}:")
        for item in outline:
            print(f"  {item['level']}: {item['text']} (Page {item['page']})")
        # Dynamically extract frequent, meaningful words from headings for bonus scoring
        from collections import Counter
        heading_words = Counter()
        for h in outline:
            words = [w.lower() for w in re.findall(r"\b\w+\b", h['text']) if len(w) > 3]
            heading_words.update(words)
        thematic_keywords = set([w for w, _ in heading_words.most_common(8)])
        candidate_sections = []
        # Dynamically detect city-specific headings using frequent proper nouns
        proper_nouns = Counter()
        for h in outline:
            for w in h['text'].split():
                if w.istitle() and len(w) > 2:
                    proper_nouns[w] += 1
        # Use top 8 proper nouns as city candidates
        city_candidates = set([w for w, _ in proper_nouns.most_common(8)])
        for h in outline:
            word_count = len(h['text'].split())
            char_count = len(h['text'])
            is_concise = 2 <= word_count <= 8 and char_count < 80
            is_thematic = any(kw in h['text'].lower() for kw in thematic_keywords)
            # City-specific if contains colon or a frequent proper noun
            is_city_specific = ':' in h['text'] or any(city in h['text'] for city in city_candidates)
            bonus = 0
            if is_concise and is_thematic and not is_city_specific:
                bonus += 10
            elif is_concise and not is_city_specific:
                bonus += 5
            elif is_thematic and not is_city_specific:
                bonus += 3
            # Penalize verbose headings and city-specific
            if word_count > 8 or char_count > 80:
                bonus -= 5
            if is_city_specific:
                bonus -= 8
            candidate_sections.append({"heading": h, "bonus": bonus})
        # Remove duplicate headings (by text)
        seen_titles = set()
        unique_candidates = []
        for c in candidate_sections:
            t = c["heading"]["text"].strip().lower()
            if t not in seen_titles:
                unique_candidates.append(c)
                seen_titles.add(t)
        unique_candidates.sort(key=lambda x: (-x["bonus"], x["heading"]["page"]))
        best_headings = [c["heading"] for c in unique_candidates if c["bonus"] > 0]
        sections = extract_sections_from_outline(pdf_path, best_headings if best_headings else [h for h in outline if h['level'] == 'H1'] or outline)
        keywords = set()
        for v in persona.values():
            keywords.update(str(v).lower().split())
        for v in job.values():
            keywords.update(str(v).lower().split())
        for sec in sections:
            raw_title = sec['title']
            title = raw_title.strip()
            title = re.sub(r'^[^A-Za-z0-9]+', '', title)
            title = re.sub(r'[:.,;\-]+$', '', title)
            title = re.sub(r'\s+', ' ', title)
            # If title is too long or looks like a sentence, try to extract first phrase
            if len(title.split()) > 12 or title.endswith('.'):
                title = title.split('.')[0].strip()
                if len(title.split()) > 12:
                    title = ' '.join(title.split()[:12]) + '...'
            # If title is still too short, skip
            if len(title) < 5:
                continue
            content = sec.get('content', '')
            title_words = set(title.lower().split())
            content_words = set(content.lower().split())
            # Score: +5 for travel keyword in title, +2 for persona/job keyword in title, +1 for persona/job keyword in content
            score = 0
            for word in travel_keywords:
                if word in title_words:
                    score += 5
            for word in keywords:
                if word in title_words:
                    score += 2
                if word in content_words:
                    score += 1
            # Penalize sentence-like/fragment titles
            if len(title) < 8 or len(title.split()) < 2:
                score -= 3
            if len(title) > 80:
                score -= 2
            if title.strip().lower() in {"conclusion", "introduction", "summary", "abstract", "table of contents"}:
                score -= 10
            if re.match(r"^chapter|^section", title.strip().lower()):
                score -= 6
            # Penalize if title looks like a sentence (starts with uppercase, rest lowercase, ends with .)
            if re.match(r'^[A-Z][a-z]+.*\.$', title):
                score -= 5
            section_obj = {
                'document': doc['filename'],
                'section_title': title,
                'score': score,
                'page_number': sec['page'],
                'pdf_path': pdf_path
            }
            all_sections.append(section_obj)
            section_objs_by_doc_page[(doc['filename'], title, sec['page'])] = section_obj
    # Sort all sections globally by score (descending), then by page number (ascending), then prefer diversity
    all_sections_sorted = sorted(all_sections, key=lambda x: (-x['score'], x['page_number'], x['document']))
    # Prefer diversity: pick at most one section per document, unless not enough unique docs
    seen_docs = set()
    diverse_sections = []
    for sec in all_sections_sorted:
        if sec['document'] not in seen_docs:
            diverse_sections.append(sec)
            seen_docs.add(sec['document'])
        if len(diverse_sections) == 5:
            break
    # If less than 5, fill up with next best regardless of doc
    if len(diverse_sections) < 5:
        for sec in all_sections_sorted:
            if sec not in diverse_sections:
                diverse_sections.append(sec)
            if len(diverse_sections) == 5:
                break
    top_sections = diverse_sections[:5]
    # Assign importance_rank 1-5
    for i, sec in enumerate(top_sections):
        sec['importance_rank'] = i + 1
        del sec['score']
    # Only keep subsection_analysis for those top 5, and generate summary for each
    top_subsections = []
    for sec in top_sections:
        summary = extract_refined_subsections(sec)
        top_subsections.append({
            'document': sec['document'],
            'refined_text': summary,
            'page_number': sec['page_number']
        })
    output = generate_output_json(input_docs, persona, job, top_sections, top_subsections)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()