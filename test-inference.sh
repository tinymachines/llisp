#!/bin/bash

# MNIST OCR Inference Test Script
# This script loads the MNIST OCR system and runs inference tests

echo "=== MNIST OCR Inference Test ==="
echo ""

# Check if SBCL is available
if ! command -v sbcl &> /dev/null; then
    echo "Error: SBCL not found. Please install SBCL first."
    echo "On Ubuntu/Debian: sudo apt install sbcl"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: data/ directory not found."
    echo "Please download MNIST data first:"
    echo ""
    echo "mkdir -p data"
    echo "cd data"
    echo "wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    echo "wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    echo "wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    echo "wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    echo "gunzip *.gz"
    echo "cd .."
    exit 1
fi

# Check if required files exist
required_files=(
    "data/train-images-idx3-ubyte"
    "data/train-labels-idx1-ubyte"
    "data/t10k-images-idx3-ubyte"
    "data/t10k-labels-idx1-ubyte"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file $file not found."
        echo "Please download and extract MNIST data files."
        exit 1
    fi
done

echo "✓ SBCL found"
echo "✓ Data directory exists"
echo "✓ MNIST files found"
echo ""

# Check if system can be loaded
echo "Testing system load..."
sbcl --eval "(progn (require :asdf) (push #P\"./\" asdf:*central-registry*) (asdf:load-system :mnist-ocr) (quit))" \
     --non-interactive 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ MNIST OCR system loads successfully"
else
    echo "✗ Error loading MNIST OCR system"
    echo "Trying to load system with error output..."
    sbcl --eval "(progn (require :asdf) (push #P\"./\" asdf:*central-registry*) (asdf:load-system :mnist-ocr) (quit))"
    exit 1
fi

echo ""
echo "Starting inference test..."
echo "=========================================="

# Run the inference test
sbcl --eval "(require :asdf)" \
     --load mnist-ocr.asd \
     --eval "(asdf:load-system :mnist-ocr)" \
     --eval "(load \"test-inference.lisp\")" \
     --eval "(test-inference)" \
     --eval "(quit)"

echo ""
echo "=========================================="
echo "Test completed!"