# Challenge 1B: Persona-Driven Document Intelligence

## Overview

This solution implements a sophisticated persona-driven document intelligence system that extracts and prioritizes the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## 🎯 Problem Statement

Build a system that acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on:
- **Persona Definition**: Role description with specific expertise and focus areas
- **Job-to-be-Done**: Concrete task the persona needs to accomplish
- **Document Collection**: 3-10 related PDFs from any domain

## 🏗️ Architecture

### Core Components

1. **Enhanced PDF Parser** (`enhanced_parser.py`)
   - Extracts hierarchical outlines (H1, H2, H3) from PDFs
   - Filters out body text and focuses on meaningful headings
   - Uses font analysis and pattern matching for better accuracy

2. **Persona-Driven Selector** (`selector.py`)
   - Hybrid selection approach combining keyword matching and AI analysis
   - Calculates relevance scores based on persona and job requirements
   - Generates comprehensive output with section justifications

3. **Automated Runner** (`run_challenge1b.py`)
   - End-to-end processing pipeline
   - Handles PDF parsing and section selection
   - Validates output and provides detailed feedback

4. **Test Framework** (`test_solution.py`)
   - Comprehensive validation of the entire system
   - Tests multiple collections and scenarios
   - Generates detailed reports

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
python setup.py

# Or manually:
pip install -r requirements.txt
```

### 2. Set OpenAI API Key (Optional but Recommended)

```bash
# Windows
set OPENAI_API_KEY=your_key_here

# Linux/Mac
export OPENAI_API_KEY=your_key_here
```

### 3. Run the Solution

```bash
# Test all collections
python test_solution.py

# Process specific collection
python run_challenge1b.py "Collection 1"

# Parse PDFs only
python run_challenge1b.py "Collection 1" --parse-only

# Select sections only (if PDFs already parsed)
python run_challenge1b.py "Collection 1" --select-only
```

## How to Use
- Place your PDFs in each collection's `PDFs/` folder.
- Edit `challenge1b_input.json` in each collection to list the PDFs, persona, and job.
- Run your analysis script in each collection folder to generate `challenge1b_output.json`.

## Input/Output Format
See the official repo and sample input/output JSONs in each collection folder.
