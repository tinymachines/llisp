# 🤖 Deep Learning from Scratch in Common Lisp

Two complete implementations of neural architectures built entirely from scratch in Common Lisp with SBCL:

1. **MNIST OCR Neural Network** - A feedforward neural network for handwritten digit recognition
2. **Transformer Model** - A sequence-to-sequence transformer that learns to add 2-digit numbers

No external ML libraries required!

## 🚀 Features

- **Pure Common Lisp** - Everything implemented from scratch
- **Educational Focus** - 101-style code with clear explanations
- **SBCL Optimized** - High-performance numerical computing
- **Interactive Development** - REPL-driven machine learning
- **Complete Pipeline** - Data loading → Training → Inference
- **High Accuracy** - Achieves >97% accuracy on MNIST test set

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

### Option 1: Direct Download from Google Storage (Recommended)
```bash
cd data/

# Download and extract all files
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# Extract all files
gunzip *.gz
```

### Option 2: Alternative Download (if above fails)
```bash
cd data/

# The original Yann LeCun site
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
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

## 🏃‍♂️ Quick Start for First-Time LISP Users

### Step 1: Start SBCL REPL
Open your terminal and type:
```bash
sbcl
```

You should see something like:
```
This is SBCL 2.1.11, an implementation of ANSI Common Lisp.
More information about SBCL is available at <http://www.sbcl.org/>.
...
* 
```

The `*` is the REPL prompt where you type commands.

### Step 2: Load the Project Files

Copy and paste these commands ONE AT A TIME into the REPL:

```lisp
;; Load individual source files (since ASDF might not be configured)
(load "packages.lisp")
```
Press Enter. You should see `T` (meaning success).

```lisp
(load "src/data-loader.lisp")
```
Press Enter. You should see `T`.

```lisp
(load "src/neural-net.lisp")
```
Press Enter. You should see `T`.

```lisp
(load "src/training.lisp")
```
Press Enter. You should see `T`.

```lisp
(load "src/inference.lisp")
```
Press Enter. You should see `T`.

### Step 3: Load the MNIST Data

Copy and paste this command:
```lisp
;; Load the MNIST dataset from the data/ directory
(multiple-value-bind (train-images train-labels test-images test-labels)
    (mnist-ocr::load-mnist-data "data/")
  ;; Store them in global variables for easy access
  (defparameter *train-images* train-images)
  (defparameter *train-labels* train-labels)  
  (defparameter *test-images* test-images)
  (defparameter *test-labels* test-labels)
  (format t "Loaded ~D training and ~D test images~%" 
          (length train-images) (length test-images)))
```

You should see:
```
Loading MNIST data from data/
Loaded 60000 training and 10000 test images
```

### Step 4: Create a Neural Network

Copy and paste:
```lisp
;; Create a neural network with:
;; - 784 inputs (28x28 pixels)
;; - 128 hidden neurons
;; - 10 outputs (digits 0-9)
(defparameter *network* (mnist-ocr::create-network 784 128 10))
(format t "Network created!~%")
```

### Step 5: Test Inference (Before Training)

Let's test the network on a few images to see random predictions:
```lisp
;; Test on first 5 images (network is untrained, so predictions are random)
(dotimes (i 5)
  (let* ((image (aref *test-images* i))
         (actual-label (aref *test-labels* i))
         (predicted (mnist-ocr::predict-digit *network* image)))
    (format t "Image ~D: Actual=~D, Predicted=~D~%" 
            i actual-label predicted)))
```

You'll see random predictions since the network isn't trained yet.

### Step 6: Train the Network (Quick Version)

For a quick test with just 1000 images and 3 epochs:
```lisp
;; Quick training (takes ~30 seconds)
(mnist-ocr::train-network *network* 
                          (subseq *train-images* 0 1000)  ; First 1000 images
                          (subseq *train-labels* 0 1000)  ; First 1000 labels
                          3                               ; 3 epochs
                          0.1)                            ; Learning rate
```

You'll see output like:
```
Epoch 0/3...
Batch 100/1000
...
Epoch 0: Loss = 0.8234, Training Accuracy = 72.50%
```

### Step 7: Test the Trained Network

```lisp
;; Test on the same 5 images again
(dotimes (i 5)
  (let* ((image (aref *test-images* i))
         (actual-label (aref *test-labels* i))
         (predicted (mnist-ocr::predict-digit *network* image)))
    (format t "Image ~D: Actual=~D, Predicted=~D ~A~%" 
            i actual-label predicted
            (if (= actual-label predicted) "✓" "✗"))))
```

### Step 8: Evaluate Overall Accuracy

```lisp
;; Test on first 1000 test images
(mnist-ocr::evaluate-accuracy *network* 
                              (subseq *test-images* 0 1000)
                              (subseq *test-labels* 0 1000))
```

### Step 9: Save Your Trained Network

```lisp
;; Save the network to a file
(mnist-ocr::save-network *network* "my-trained-network.lisp")
```

### Step 10: Exit SBCL

Type:
```lisp
(quit)
```

## 🚀 Full Training (Advanced)

Once you're comfortable with the basics, train on the full dataset:

```lisp
;; Full training (takes 5-10 minutes)
(mnist-ocr::train-network *network* 
                          *train-images*   ; All 60000 images
                          *train-labels*   ; All 60000 labels
                          10               ; 10 epochs
                          0.1)             ; Learning rate

;; Evaluate on full test set
(mnist-ocr::evaluate-accuracy *network* *test-images* *test-labels*)
```

## 💡 Understanding What Each Command Does

### Loading Files
- `(load "packages.lisp")` - Defines the namespace for our code
- `(load "src/data-loader.lisp")` - Functions to read MNIST binary files
- `(load "src/neural-net.lisp")` - Neural network structure and forward pass
- `(load "src/training.lisp")` - Backpropagation and training loop
- `(load "src/inference.lisp")` - Prediction and evaluation functions

### Key Functions
- `load-mnist-data` - Reads the MNIST files and returns 4 arrays
- `create-network` - Initializes random weights for the neural network
- `predict-digit` - Takes an image array and returns predicted digit (0-9)
- `train-network` - Updates weights using gradient descent
- `evaluate-accuracy` - Tests the network and reports percentage correct

### Data Structures
- Images are arrays of 784 bytes (28×28 pixels)
- Labels are single bytes (0-9)
- Network has weight matrices and bias vectors
- All computations use single-precision floats for speed


## 🎯 Common Usage Patterns

### Loading a Previously Trained Network
```lisp
;; Start SBCL and load the system
(load "packages.lisp")
(load "src/data-loader.lisp") 
(load "src/neural-net.lisp")
(load "src/training.lisp")
(load "src/inference.lisp")

;; Load saved network
(defparameter *saved-net* (mnist-ocr::load-network "trained-mnist-net.lisp"))

;; Load test data to try predictions
(multiple-value-bind (tr-img tr-lbl test-images test-labels)
    (mnist-ocr::load-mnist-data "data/")
  (defparameter *test-images* test-images)
  (defparameter *test-labels* test-labels))

;; Make predictions
(mnist-ocr::predict-digit *saved-net* (aref *test-images* 0))
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

### Common REPL Errors and Solutions

**"Package MNIST-OCR does not exist"**
- You forgot to load packages.lisp first
- Solution: `(load "packages.lisp")`

**"Undefined function MNIST-OCR::..."**  
- You haven't loaded all the source files
- Solution: Load all files in order as shown in Step 2

**"File not found" errors**
- You're not in the project directory
- Solution: Check with `(pwd)` or restart SBCL from project folder

**"END-OF-FILE" errors**
- MNIST data files are corrupted or not fully downloaded
- Solution: Re-download and extract the .gz files

### Memory Issues
```lisp
;; Force garbage collection
(sb-ext:gc :full t)

;; Check memory usage
(room)

;; Use smaller batches for training
(subseq *train-images* 0 5000)  ; Just 5000 images
```

### Performance Tips
- Training 60000 images takes 5-10 minutes
- Start with 1000 images for quick tests
- The network should achieve >90% accuracy after 3-5 epochs

### ASDF Not Working?
No problem! Just use the manual loading approach shown in the first-time user section.

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

## 🤖 Transformer Implementation

In addition to the MNIST neural network, this repository includes a complete transformer implementation inspired by the tinygrad example.

### Transformer Features

- **Multi-Head Attention** - Scaled dot-product attention with multiple heads
- **Positional Encoding** - Learnable position embeddings
- **Layer Normalization** - For stable training
- **Feed-Forward Networks** - With ReLU activation
- **Residual Connections** - Skip connections for better gradient flow
- **Addition Task** - Learns to add 2-digit numbers (e.g., 35 + 47 = 082)

### Quick Start - Transformer

```bash
# Start SBCL
sbcl
```

In the REPL:
```lisp
;; Add current directory to ASDF
(push #P"./" asdf:*central-registry*)

;; Load the transformer system
(asdf:load-system :transformer)

;; Switch to transformer package
(in-package :transformer)

;; Run the addition experiment
(main)
```

This will:
1. Create a transformer with 2 layers, 4 attention heads
2. Generate a dataset of 10,000 addition problems
3. Train for 10 epochs with learning rate decay
4. Show test examples like: `35 + 47 = 082 (predicted: 082)`

### Transformer Architecture

```
Input (6 tokens: d1 d2 + d3 d4 =)
         ↓
    Embedding Layer
         ↓
  Positional Encoding
         ↓
  Transformer Block 1
   - Multi-Head Attention
   - Feed-Forward Network
         ↓
  Transformer Block 2
   - Multi-Head Attention  
   - Feed-Forward Network
         ↓
   Output Projection
         ↓
Output (6 tokens: d2 + d3 d4 = result)
```

### Understanding the Code

**Core Components:**
- `src/attention.lisp` - Scaled dot-product and multi-head attention
- `src/transformer-block.lisp` - Complete transformer block with residual connections
- `src/transformer-model.lisp` - Full model with embeddings and output projection
- `src/dataset.lisp` - Addition problem generator
- `src/transformer-training.lisp` - Training loop with Adam optimizer

**Key Functions:**
- `(make-instance 'transformer ...)` - Create a transformer model
- `(forward model input)` - Run forward pass
- `(train-transformer ...)` - Train the model
- `(make-addition-dataset)` - Generate training data

### Experimenting with Transformers

```lisp
;; Create a smaller transformer
(defparameter *small-transformer*
  (make-instance 'transformer
                 :num-symbols 10
                 :max-len 6
                 :num-layers 1      ; Just 1 layer
                 :embed-dim 64      ; Smaller embeddings
                 :num-heads 2       ; Fewer attention heads
                 :ff-dim 16))       ; Smaller feed-forward

;; Create a larger transformer
(defparameter *large-transformer*
  (make-instance 'transformer
                 :num-symbols 10
                 :max-len 6
                 :num-layers 4      ; More layers
                 :embed-dim 256     ; Larger embeddings
                 :num-heads 8       ; More attention heads
                 :ff-dim 128))      ; Larger feed-forward
```

## 🎉 What's Next?

Once you have both models working, try:
- Implementing other transformer tasks (copy, reverse, sort)
- Adding beam search for better predictions
- Implementing BERT-style masked language modeling
- Creating a web interface with Hunchentoot
- Building a REST API for both models
- Experimenting with other sequence tasks

Happy Lisping! 🎊
