# 🤖 MNIST OCR from Scratch in Common Lisp

A complete implementation of a neural network for MNIST digit recognition built entirely from scratch in Common Lisp with SBCL. No external ML libraries required!

## 🚀 Features

- **Pure Common Lisp** - Everything implemented from scratch
- **Educational Focus** - 101-style code with clear explanations
- **SBCL Optimized** - High-performance numerical computing
- **Interactive Development** - REPL-driven machine learning
- **Complete Pipeline** - Data loading → Training → Inference

## 📋 Prerequisites

### Install SBCL (Steel Bank Common Lisp)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install sbcl
```

**macOS (using Homebrew):**
```bash
brew install sbcl
```

**Arch Linux:**
```bash
sudo pacman -S sbcl
```

**From Source:**
```bash
# Download from http://www.sbcl.org/platform-table.html
wget http://www.sbcl.org/platform-table.html
# Follow platform-specific instructions
```

### Verify Installation
```bash
sbcl --version
# Should output something like: SBCL 2.3.x
```

## 📁 Project Setup

### 1. Clone or Create Project Directory
```bash
mkdir mnist-ocr-lisp
cd mnist-ocr-lisp
```

### 2. Create Project Structure
```bash
mkdir -p src examples data
touch mnist-ocr.asd packages.lisp
touch src/{data-loader,neural-net,training,inference}.lisp
touch examples/train-mnist.lisp
```

Your directory should look like:
```
mnist-ocr-lisp/
├── mnist-ocr.asd           # ASDF system definition
├── packages.lisp           # Package definitions  
├── README.md              # This file
├── src/
│   ├── data-loader.lisp   # MNIST file parsing
│   ├── neural-net.lisp    # Neural network implementation
│   ├── training.lisp      # Training algorithms
│   └── inference.lisp     # Prediction functions
├── examples/
│   └── train-mnist.lisp   # Complete training example
└── data/                  # MNIST dataset files (see below)
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte  
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

## 📊 Download MNIST Dataset

### Option 1: Direct Download (Recommended)
```bash
cd data/

# Download training images
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gunzip train-images-idx3-ubyte.gz

# Download training labels  
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip train-labels-idx1-ubyte.gz

# Download test images
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz

# Download test labels
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  
gunzip t10k-labels-idx1-ubyte.gz
```

### Option 2: One-liner Download Script
```bash
cd data/
curl -s http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz} | gunzip -c > mnist-files.tmp
# Note: You'll need to extract these manually or use the individual commands above
```

### Verify Dataset Files
```bash
ls -la data/
# Should show ~47MB train-images, ~60KB train-labels, ~7.8MB test images, ~10KB test labels
```

## 🛠️ Installation & Build

### 1. Create ASDF System Definition
Create `mnist-ocr.asd`:
```lisp
(asdf:defsystem "mnist-ocr"
  :description "MNIST OCR from scratch in Common Lisp"
  :author "Your Name <your.email@example.com>"
  :license "MIT"
  :version "1.0.0"
  :serial t
  :components ((:file "packages")
               (:file "src/data-loader")
               (:file "src/neural-net") 
               (:file "src/training")
               (:file "src/inference")
               (:file "examples/train-mnist")))
```

### 2. Define Packages
Create `packages.lisp`:
```lisp
(defpackage #:mnist-ocr
  (:use #:cl)
  (:export #:load-mnist-data
           #:create-network
           #:train-network
           #:predict-digit
           #:evaluate-accuracy
           #:save-network
           #:load-network))
```

### 3. Load the System
```bash
sbcl
```

In the SBCL REPL:
```lisp
(require :asdf)
(asdf:load-system :mnist-ocr)
```

## 🏃‍♂️ Quick Start

### 1. Start SBCL and Load System
```bash
sbcl --load mnist-ocr.asd
```

### 2. In the REPL, Train Your First Network
```lisp
;; Load the system
(asdf:load-system :mnist-ocr)

;; Switch to our package
(in-package :mnist-ocr)

;; Load MNIST data
(multiple-value-bind (train-images train-labels test-images test-labels)
    (load-mnist-data "data/")
  
  ;; Create a simple network: 784 inputs -> 128 hidden -> 10 outputs
  (let ((network (create-network 784 128 10)))
    
    ;; Train for 10 epochs
    (format t "Starting training...~%")
    (train-network network train-images train-labels 10 0.1)
    
    ;; Test the network
    (format t "Evaluating on test set...~%")
    (evaluate-accuracy network test-images test-labels)
    
    ;; Save the trained network
    (save-network network "trained-mnist-net.lisp")))
```

### 3. Expected Output
```
Starting training...
Epoch 0: Loss = 2.1234, Accuracy = 23.45%
Epoch 1: Loss = 1.8901, Accuracy = 45.67%
Epoch 2: Loss = 1.5678, Accuracy = 65.43%
...
Epoch 9: Loss = 0.4567, Accuracy = 89.12%

Evaluating on test set...
Test Accuracy: 87.65%
Network saved to trained-mnist-net.lisp
```

## 🎯 Usage Examples

### Predict Single Digits
```lisp
;; Load a trained network
(defparameter *my-network* (load-network "trained-mnist-net.lisp"))

;; Predict a single image (returns digit 0-9)
(let ((first-test-image (aref test-images 0)))
  (format t "Predicted digit: ~A~%" 
          (predict-digit *my-network* first-test-image)))

;; Batch predictions
(dotimes (i 10)
  (format t "Image ~A: Predicted ~A, Actual ~A~%" 
          i 
          (predict-digit *my-network* (aref test-images i))
          (aref test-labels i)))
```

### Experiment with Hyperparameters
```lisp
;; Try different learning rates
(train-network network train-images train-labels 5 0.01)   ; Slower learning
(train-network network train-images train-labels 5 0.5)    ; Faster learning

;; Try different network architectures  
(defparameter *small-net* (create-network 784 64 10))      ; Smaller hidden layer
(defparameter *large-net* (create-network 784 256 10))     ; Larger hidden layer
```

### Interactive Debugging
```lisp
;; Inspect network weights
(describe *my-network*)

;; Watch training progress step by step
(setf *print-training-progress* t)

;; Test on individual examples
(let* ((image (aref test-images 42))
       (normalized (normalize-image image))
       (output (forward-pass *my-network* normalized)))
  (format t "Raw outputs: ~A~%" output)
  (format t "Predicted class: ~A~%" (argmax output)))
```

## ⚡ Performance Tips

### 1. Compile for Speed
Add to the top of your files:
```lisp
(declaim (optimize (speed 3) (safety 0) (debug 0)))
```

### 2. Use Smaller Dataset for Testing
```lisp
;; Use only first 1000 examples for quick experiments
(train-network network 
               (subseq train-images 0 1000)
               (subseq train-labels 0 1000) 
               5 0.1)
```

### 3. Monitor Memory Usage
```lisp
;; Check memory usage
(sb-ext:gc :full t)
(room)
```

## 🐛 Troubleshooting

### "File not found" errors
- Ensure MNIST files are in the `data/` directory
- Check file names match exactly (no extra extensions)
- Verify files were unzipped properly

### Memory issues
- Use smaller batch sizes
- Reduce training set size for initial experiments
- Run `(sb-ext:gc :full t)` to force garbage collection

### Slow training
- Ensure optimization declarations are included
- Use single-float instead of double-float
- Consider smaller network architecture for initial tests

### ASDF system not found
```lisp
;; Add current directory to ASDF
(push #P"./" asdf:*central-registry*)
(asdf:load-system :mnist-ocr)
```

## 📈 Extending the Project

### Add More Features
- Implement momentum in gradient descent
- Add dropout regularization  
- Try different activation functions (ReLU, tanh)
- Implement batch normalization
- Add convolutional layers

### Visualization
- Plot training loss over time
- Visualize learned weights as images
- Show misclassified examples
- Create confusion matrices

### Advanced Architectures
- Multi-layer networks (deep learning)
- Convolutional Neural Networks (CNNs)
- Recurrent networks for sequence data

## 📚 Learning Resources

- **SBCL Manual**: http://www.sbcl.org/manual/
- **Common Lisp Cookbook**: https://lispcookbook.github.io/cl-cookbook/
- **Neural Networks from Scratch**: Understanding backpropagation step-by-step
- **MNIST Database**: http://yann.lecun.com/exdb/mnist/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎉 What's Next?

Once you have this working, try:
- Implementing a web interface with Hunchentoot
- Creating a GUI with McCLIM  
- Deploying as a command-line tool
- Building a REST API for digit recognition
- Experimenting with other datasets (CIFAR-10, Fashion-MNIST)

Happy Lisping! 🎊
