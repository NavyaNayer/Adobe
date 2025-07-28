# Docker Instructions for Challenge 1B - Persona-Driven Document Intelligence

## Building the Docker Image

**Important**: Run these commands from the `Challenge_1b` directory.

```bash
cd Challenge_1b
docker build --platform linux/amd64 -t challenge1b:latest .
```

## Running the Container

### Option 1: Single Collection Processing
For processing a single collection with its input file:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1b:latest
```

Expected input structure:
```
input/
├── challenge1b_input.json
└── PDFs/
    ├── document1.pdf
    ├── document2.pdf
    └── ...
```

### Option 2: Multiple Collections Processing
For processing multiple collections:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1b:latest
```

Expected input structure:
```
input/
├── Collection_A/
│   ├── challenge1b_input.json
│   └── PDFs/
├── Collection_B/
│   ├── challenge1b_input.json
│   └── PDFs/
└── ...
```

## What the Container Does

1. **Processes PDF collections** from `/app/input` directory
2. **Extracts PDF outlines** using enhanced Challenge 1A extractor
3. **Runs persona-driven selection** based on input requirements
4. **Generates intelligent outputs** in `/app/output` directory

## Input File Format

Each collection needs a `challenge1b_input.json` file:

```json
{
  "persona": {
    "role": "Travel Planner",
    "expertise": "Tourism, Logistics",
    "preferences": "Practical information"
  },
  "job_to_be_done": {
    "task": "Plan a trip",
    "constraints": "4 days, 10 people",
    "goals": "Maximize experiences"
  },
  "documents": ["doc1.pdf", "doc2.pdf"]
}
```

## Output Format

Generates `challenge1b_output.json` (or `CollectionName_output.json` for multiple collections):

```json
{
  "metadata": {
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip...",
    "processing_timestamp": "2025-07-28T..."
  },
  "extracted_sections": [
    {
      "document": "document.pdf",
      "section_title": "Relevant Section",
      "page_number": 5,
      "relevance_score": 0.95,
      "reasoning": "High relevance because..."
    }
  ]
}
```

## Requirements Met

- ✅ **AMD64 Architecture**: Uses `--platform=linux/amd64`
- ✅ **No GPU Dependencies**: CPU-only processing
- ✅ **Offline Operation**: Works with `--network none`
- ✅ **Automatic Processing**: Handles single or multiple collections
- ✅ **Intelligent Selection**: Persona-driven document intelligence

## Features

- ✅ **Enhanced PDF Processing**: Inherits table detection from Challenge 1A
- ✅ **Persona-Driven AI**: Intelligent section selection based on user profiles
- ✅ **Multi-Collection Support**: Batch processing of document collections
- ✅ **Offline ML**: Uses scikit-learn for semantic similarity (no network calls)
- ✅ **Robust Error Handling**: Graceful failure handling and detailed logging
