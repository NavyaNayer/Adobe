# Challenge 1B: Persona-Driven Document Intelligence
## Adobe India Hackathon 2025 - "Connecting the Dots"

### Complete Implementation

Built an **advanced persona-driven document intelligence system** that extracts and prioritizes relevant sections from document collections based on specific personas and their job-to-be-done, using cutting-edge semantic similarity and hybrid AI analysis.

## Key Achievements

✅ **Intelligent Document Analysis**: Automatically extracts and ranks relevant sections based on personas and job requirements  
✅ **Hybrid AI Engine**: Combines sentence-transformers semantic similarity with TF-IDF fallback and keyword matching  
✅ **Cross-Domain Intelligence**: Handles academic, business, technical, culinary content without hardcoding  
✅ **Performance Optimized**: CPU-only, <1GB, <60-second processing with Docker deployment  
✅ **Production Ready**: Offline execution, robust error handling, comprehensive dependency management

## Technical Implementation

**Semantic Analysis**: Deployed sentence-transformers with sklearn TF-IDF fallback for robust semantic understanding  
**Domain-Aware Processing**: Automatic domain detection with specialized pipelines maintaining generalizability  
**Adaptive Scoring**: Dynamic weighting between semantic similarity and keyword matching based on confidence  
**Multi-Strategy Extraction**: 7-layer fallback system for reliable section identification across diverse PDF formats  
**Intelligent Filtering**: Requirement-aware filtering for dietary, academic, technical, and business contexts

## Docker Deployment

**Platform**: AMD64, CPU-only, <1GB, offline execution

```bash
# Build and run the solution
docker build --platform linux/amd64 -t challenge1b:latest .
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1b:latest
```

**Behavior**: Processes PDFs from `/app/input`, generates `filename.json` in `/app/output`

## Architecture & Components

1. **PersonaDrivenSelector** (`selector.py`): Hybrid AI engine with semantic similarity and domain-specific keyword matching
2. **Semantic Engine**: sentence-transformers with sklearn TF-IDF fallback for universal compatibility  
3. **Processing Pipeline**: Challenge 1A integration with multi-language support and intelligent filtering
4. **Pipeline Runner** (`run_challenge1b.py`): End-to-end automation with dependency management and validation

**Innovations**: Hybrid semantic similarity, multi-domain processing, explainable AI rankings, batch optimization, confidence-based scoring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers  # optional for enhanced performance

# Run the system
python run_challenge1b.py "Collection 1"          # Full pipeline
python run_challenge1b.py "Collection 1" --parse-only    # PDF parsing only
python run_challenge1b.py "Collection 1" --select-only   # Analysis only
python selector.py "Collection 1"                 # Direct selector
```

**Collection Structure**: Place PDFs in `PDFs/` folder, edit `challenge1b_input.json`, run analysis to generate `challenge1b_output.json`

## Input/Output Format

**Input** (`challenge1b_input.json`):
```json
{
  "persona": {"role": "Professional Chef specializing in Mediterranean cuisine"},
  "job_to_be_done": {"task": "Find vegetarian recipes suitable for dinner party"},
  "documents": [{"filename": "mediterranean_cookbook.pdf"}]
}
```

**Output** (`challenge1b_output.json`):
```json
{
  "metadata": {
    "persona": "HR professional",
    "job_to_be_done": "Create and manage fillable forms",
    "method": "semantic similarity"
  },
  "extracted_sections": [
    {"document": "forms.pdf", "page_number": 0, "section_title": "Fill and Sign PDF Forms", "importance_rank": 1}
  ],
  "subsection_analysis": [
    {"document": "forms.pdf", "section_title": "Fill and Sign PDF Forms", "refined_text": "You can easily fill, sign..."}
  ]
}
```

## Evaluation Success

| Criteria | Achievement |
|----------|-------------|
| **Section Relevance (60 pts)** | ✅ Advanced semantic similarity + keyword matching with transparent ranking |
| **Sub-Section Relevance (40 pts)** | ✅ Multi-strategy extraction with intelligent filtering and pattern recognition |
| **Generic Solution** | ✅ Handles unlimited domains, personas, job types without hardcoding |
| **Technical Constraints** | ✅ CPU-only, offline, <60s, <1GB compliant with Docker deployment |

## Dependencies

**Core**: PyMuPDF, langdetect, numpy, scikit-learn  
**Enhanced**: sentence-transformers, transformers, torch (optional)

```bash
pip install PyMuPDF langdetect numpy scikit-learn
pip install sentence-transformers torch  # for enhanced performance
```

## Author

Team Smart Builders
-Diksha Khandelwal
-Manvendra Singh Tanwar
-Navya Nayer



---
*Submission for Adobe India Hackathon 2025 - "Connecting the Dots" Challenge*
