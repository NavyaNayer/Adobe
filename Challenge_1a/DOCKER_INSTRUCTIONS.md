# Docker Instructions for PDF Outline Extractor - Challenge 1A

## Building the Docker Image

**Important**: Run these commands from the `Challenge_1a` directory.

```bash
cd Challenge_1a
docker build --no-cache --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

## Running the Container

Run the container to process PDFs from input directory and generate JSON files in output directory:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

## What the Container Does

1. **Processes all PDF files** from the `/app/input` directory
2. **Generates corresponding JSON files** in `/app/output` directory
   - For each `filename.pdf`, creates `filename.json`
3. **Uses enhanced PDF extraction** with:
   - Table detection and exclusion
   - Enhanced title detection
   - Corruption pattern detection
   - Multi-language support
4. **Works completely offline** - no network calls required

## Output Format

Each generated JSON file contains:
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Section Title",
      "page": 0
    }
  ]
}
```

## Requirements Met

- ✅ **AMD64 Architecture**: Uses `--platform=linux/amd64`
- ✅ **No GPU Dependencies**: CPU-only processing
- ✅ **Small Model Size**: PyMuPDF (~50MB) + langdetect (~1MB)
- ✅ **Offline Operation**: `--network none` compatible
- ✅ **Automatic Processing**: Processes all PDFs in input directory
- ✅ **Correct Output**: Generates filename.json for each filename.pdf

## Example Usage

```bash
# Create input directory with PDF files
mkdir -p input output
cp your_pdfs/*.pdf input/

# Build the image
docker build --platform linux/amd64 -t pdf-extractor:v1 .

# Run the extraction
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-extractor:v1

# Check results
ls output/
```
