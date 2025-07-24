# Challenge 1b: Multi-Collection PDF Analysis

This project processes multiple document collections and extracts relevant content based on specific personas and use cases.

## Structure

Challenge_1b/
├── Collection 1/  # Travel Planning
│   ├── PDFs/  # South of France guides
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection 2/  # Adobe Acrobat Learning
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection 3/  # Recipe Collection
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
└── README.md

## How to Use
- Place your PDFs in each collection's `PDFs/` folder.
- Edit `challenge1b_input.json` in each collection to list the PDFs, persona, and job.
- Run your analysis script in each collection folder to generate `challenge1b_output.json`.

## Input/Output Format
See the official repo and sample input/output JSONs in each collection folder.
