# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Common Lisp educational project for implementing an MNIST OCR (Optical Character Recognition) system from scratch. The project builds a neural network without external ML libraries using SBCL (Steel Bank Common Lisp).

## Key Commands

### SBCL and ASDF Commands
- **Start SBCL REPL**: `sbcl` or `sbcl --load mnist-ocr.asd`
- **Load system in REPL**: `(asdf:load-system :mnist-ocr)`
- **Switch to project package**: `(in-package :mnist-ocr)`
- **Compile with optimizations**: Add `(declaim (optimize (speed 3) (safety 0) (debug 0)))` to files

### Development Commands
- **Force garbage collection**: `(sb-ext:gc :full t)`
- **Check memory usage**: `(room)`
- **Add current directory to ASDF**: `(push #P"./" asdf:*central-registry*)`

### Data Setup Commands
```bash
# Download MNIST dataset
cd data/
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

## Architecture and Structure

### Intended Project Structure
```
mnist-ocr-lisp/
├── mnist-ocr.asd               # ASDF system definition
├── packages.lisp               # Package definitions  
├── src/
│   ├── data-loader.lisp       # MNIST IDX format parsing
│   ├── neural-net.lisp        # Network architecture & forward pass
│   ├── training.lisp          # Backpropagation & optimization
│   └── inference.lisp         # Prediction & evaluation
├── examples/
│   └── train-mnist.lisp       # Complete training example
└── data/                       # MNIST dataset files
```

### Neural Network Architecture
- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer**: 128 neurons with sigmoid activation
- **Output Layer**: 10 neurons with softmax activation
- **Loss Function**: Cross-entropy
- **Training**: Mini-batch gradient descent with backpropagation

### Core Components

1. **Data Loader**: Handles MNIST IDX binary format with big-endian integers
2. **Matrix Operations**: Custom implementations for matrix multiplication, vector operations
3. **Activation Functions**: Sigmoid for hidden layer, softmax for output
4. **Backpropagation**: Chain rule implementation for gradient computation
5. **Training Loop**: Epoch-based training with loss and accuracy tracking

### SBCL Optimization Notes
- Use `(simple-array single-float (* *))` for matrices
- Declare types explicitly for numerical operations
- Use specialized arrays to eliminate bounds checking
- Apply `(declaim (optimize (speed 3)))` for performance-critical code

## Current Status
- Documentation exists (README.md, docs/FIRST-PROJECT.md)
- No source code implemented yet
- Project structure needs to be created
- MNIST data needs to be downloaded

## Testing Approach
- Interactive REPL-based testing and experimentation
- No formal testing framework currently specified
- Example usage provided for evaluating accuracy on test set