#!/bin/bash
# Test script for Docker PDF extractor

echo "🐳 Testing PDF Extractor Docker Setup"

# Create test directories
mkdir -p input output

# Check if we have any PDF files to test with
if [ ! "$(ls -A input/*.pdf 2>/dev/null)" ]; then
    echo "❌ No PDF files found for testing in input/"
    echo "Please add some PDF files to test with"
    exit 1
fi

# Copy test PDFs to input directory (they're already there)
echo "📄 Using PDFs from input directory..."

# Check if we have PDFs to test
if [ ! "$(ls -A input/*.pdf 2>/dev/null)" ]; then
    echo "❌ No PDF files copied to input directory"
    exit 1
fi

echo "📁 PDFs ready for testing:"
ls -la input/*.pdf

# Build the Docker image
echo "🔨 Building Docker image..."
docker build --platform linux/amd64 -t pdf-extractor-test:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Run the container
echo "🚀 Running PDF extraction..."
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-extractor-test:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker run failed"
    exit 1
fi

# Check results
echo "📊 Checking results..."
if [ "$(ls -A output/*.json 2>/dev/null)" ]; then
    echo "✅ JSON files generated successfully:"
    ls -la output/*.json
    
    # Show sample content
    echo ""
    echo "📄 Sample output content:"
    head -20 output/*.json | head -20
else
    echo "❌ No JSON files were generated"
    exit 1
fi

echo ""
echo "🎉 Docker setup test completed successfully!"
echo "Your Docker container is ready for submission."
