#!/bin/bash
# Test script for Challenge 1B Docker setup

echo "🧠 Testing Challenge 1B Docker Setup"

# Create test directories
mkdir -p input output

# Check if we have Collection 1 to test with
if [ ! -d "Collection 1" ]; then
    echo "❌ Collection 1 directory not found"
    echo "Please ensure you're running this from Challenge_1b directory"
    exit 1
fi

# Copy Collection 1 as test input
echo "📄 Preparing test collection..."
cp -r "Collection 1" input/

# Ensure we have the required input file
if [ ! -f "input/Collection 1/challenge1b_input.json" ]; then
    echo "❌ challenge1b_input.json not found in Collection 1"
    exit 1
fi

echo "📁 Test collection ready:"
ls -la "input/Collection 1/"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build --platform linux/amd64 -t challenge1b-test:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Run the container
echo "🚀 Running Challenge 1B processing..."
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none challenge1b-test:latest

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
    head -30 output/*.json | head -30
else
    echo "❌ No JSON files were generated"
    exit 1
fi

echo ""
echo "🎉 Challenge 1B Docker setup test completed successfully!"
echo "Your Docker container is ready for submission."
