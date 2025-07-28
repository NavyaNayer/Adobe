# Adobe India Hackathon 2025 - "Connecting the Dots"
## Complete Document Intelligence Solutions

Built **end-to-end document processing and analysis systems** for Challenge 1A (PDF Outline Extraction) and Challenge 1B (Persona-Driven Document Intelligence), delivering production-ready solutions with advanced AI capabilities.

## Challenge 1A: PDF Outline Extraction

### Key Achievements
- **Multi-Language Support**: Extracts outlines from English, French, German, Japanese, and non-Latin script PDFs
- **Robust Processing**: Handles complex document structures with intelligent heading detection
- **Format Compliance**: Generates exact JSON specification with page numbers and hierarchical sections
- **Production Ready**: Docker deployment with offline execution and comprehensive error handling

### Technical Implementation
**Advanced Text Extraction**: Multi-strategy approach with fuzzy matching and cross-page detection  
**Language Detection**: Automatic identification with specialized processing for each language  
**Heading Recognition**: Intelligent parsing of document structure and hierarchical organization

## Challenge 1B: Persona-Driven Document Intelligence

### Key Achievements
- **Intelligent Analysis**: Automatically extracts and ranks relevant sections based on personas and job requirements
- **Hybrid AI Engine**: Combines sentence-transformers semantic similarity with TF-IDF fallback and keyword matching
- **Cross-Domain Intelligence**: Handles academic, business, technical, culinary content without hardcoding
- **Performance Optimized**: CPU-only, <1GB, <60-second processing with comprehensive dependency management

### Technical Implementation
**Semantic Analysis**: Deployed sentence-transformers with sklearn TF-IDF fallback for robust understanding  
**Domain-Aware Processing**: Automatic domain detection with specialized pipelines maintaining generalizability  
**Adaptive Scoring**: Dynamic weighting between semantic similarity and keyword matching based on confidence  
**Multi-Strategy Extraction**: 7-layer fallback system for reliable section identification across diverse PDF formats

## Integrated Architecture

**Foundation (Challenge 1A)**: PDF parsing and outline extraction providing structured document analysis  
**Intelligence Layer (Challenge 1B)**: Persona-driven semantic analysis building on extracted outlines  
**Unified Pipeline**: Challenge 1B automatically integrates with Challenge 1A for end-to-end processing


**Innovations**: Multi-language PDF processing, hybrid semantic similarity, domain-aware intelligence, explainable AI rankings, integrated pipeline architecture

## Docker Deployment

**Platform**: AMD64, CPU-only, <1GB, offline execution

```bash
# Challenge 1A - PDF Outline Extraction
docker build --no-cache --platform linux/amd64 -t challenge1a:latest ./Challenge_1a
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1a:latest

# Challenge 1B - Persona-Driven Intelligence  
docker build --no-cache --platform linux/amd64 -t challenge1b:latest ./Challenge_1b
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1b:latest
```

## Quick Start

```bash
# Install dependencies for both challenges
pip install PyMuPDF langdetect numpy scikit-learn
pip install sentence-transformers torch  # enhanced performance for 1B

# Challenge 1A: Extract PDF outlines
python Challenge_1a/pdf_outline_extractor.py input/ output/

# Challenge 1B: Persona-driven analysis (auto-integrates with 1A)
python Challenge_1b/run_challenge1b.py "Collection 1"  # Full pipeline
python Challenge_1b/selector.py "Collection 1"        # Direct analysis
```

## Input/Output Integration

**Challenge 1A Input**: PDF files in `/input` directory  
**Challenge 1A Output**: JSON outlines with hierarchical structure and page numbers

**Challenge 1B Input**: Collection with PDFs + `challenge1b_input.json` (persona, job, documents)  
**Challenge 1B Output**: `challenge1b_output.json` with ranked sections and subsection analysis

**Integration**: Challenge 1B automatically uses Challenge 1A outline extraction for PDF processing

## Evaluation Success

| Challenge | Criteria | Achievement |
|-----------|----------|-------------|
| **1A** | Multi-language outline extraction | Advanced text processing with language-specific optimization |
| **1B** | Section Relevance (60 pts) | Semantic similarity + keyword matching with transparent ranking |
| **1B** | Sub-section Relevance (40 pts) | Multi-strategy extraction with intelligent filtering |
| **Both** | Technical Constraints | CPU-only, offline, <60s, <1GB compliant with Docker deployment |

## Core Dependencies

**Challenge 1A**: PyMuPDF, langdetect, numpy  
**Challenge 1B**: PyMuPDF, langdetect, numpy, scikit-learn, sentence-transformers (optional)

## Project Structure

```
Adobe/
├── Challenge_1a/          # PDF outline extraction system
│   ├── pdf_outline_extractor.py
│   ├── input/ output/     # PDF processing directories
│   ├── README.md          # Detailed Challenge 1A documentation
│   └── Dockerfile
├── Challenge_1b/          # Persona-driven intelligence system  
│   ├── selector.py        # Core intelligence engine
│   ├── run_challenge1b.py # Integrated pipeline runner
│   ├── Collection 1-3/    # Test collections
│   ├── README.md          # Detailed Challenge 1B documentation
│   └── Dockerfile
└── README.md             # This integrated documentation
```

**Note**: Each challenge folder contains its own detailed README with specific implementation details, usage instructions, and technical documentation.

## Author

Team Smart Builders
-Diksha Khandelwal
-Manvendra Singh Tanwar
-Navya Nayer


---
*Complete Submission for Adobe India Hackathon 2025 - "Connecting the Dots" Challenge*
