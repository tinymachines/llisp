# 🤖 Deep Learning from Scratch in Common Lisp

Learn deep learning by building neural networks from scratch! This repository contains standalone lessons that teach you how to implement modern AI architectures in pure Common Lisp.

## 📚 Available Lessons

### [1. MNIST Neural Network](docs/lessons/MNIST.md)
Build a feedforward neural network that recognizes handwritten digits with >95% accuracy. Learn backpropagation, gradient descent, and achieve real results on the famous MNIST dataset.

### [2. Transformer from Scratch](docs/lessons/TRANSFORMER.md) 
Implement the attention mechanism that powers ChatGPT! Build a complete transformer that learns to perform 2-digit addition, understanding multi-head attention, layer normalization, and more.

### More lessons coming soon!
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs) 
- Generative Adversarial Networks (GANs)
- Reinforcement Learning

## 🚀 Features

- **Zero Dependencies** - Pure Common Lisp, no external ML libraries
- **Copy-Paste Learning** - Each lesson is a single file you can run
- **Hands-On Code** - Learn by doing, not just reading
- **Production Techniques** - Real optimizations and best practices
- **Clear Explanations** - Understand the math and the code
- **REPL-Driven** - Interactive development and experimentation

## 📋 Prerequisites

You only need SBCL (Steel Bank Common Lisp) installed:

**Ubuntu/Debian:**
```bash
sudo apt install sbcl
```

**macOS:**
```bash
brew install sbcl
```

**Windows:**
Download from [sbcl.org](http://www.sbcl.org/platform-table.html)

## 🚀 Quick Start

1. **Pick a lesson** from the [lessons folder](docs/lessons/)
2. **Copy the code** into a file (e.g., `lesson.lisp`)
3. **Run it** with `sbcl --script lesson.lisp`

That's it! Each lesson is completely self-contained.

## 📖 How to Use This Repository

### For Beginners
Start with the MNIST lesson - it's the "Hello World" of deep learning. You'll build a neural network that recognizes handwritten digits with impressive accuracy.

### For Experienced Developers
Jump straight to the Transformer lesson to understand the architecture behind modern LLMs. The code is optimized and uses advanced Lisp techniques.

### For Teachers
Each lesson includes:
- Complete working code
- Detailed explanations
- Exercises and extensions
- Common pitfalls and solutions

## 🎯 Learning Path

1. **Start with MNIST** - Understand the basics of neural networks
2. **Move to Transformers** - Learn modern attention mechanisms
3. **Experiment** - Modify the code, try different architectures
4. **Build your own** - Create new models from scratch

## 💡 Why Common Lisp?

- **Interactive Development** - Change code while it's running
- **Fast Prototyping** - Test ideas immediately in the REPL
- **Symbolic Computing** - Natural for AI and machine learning
- **Performance** - Compiled code rivals C when optimized
- **Learning Focus** - No black-box libraries hiding the details

## 🤝 Contributing

We welcome new lessons! If you've implemented an interesting model in Common Lisp:

1. Create a self-contained lesson file
2. Add clear explanations and examples
3. Submit a pull request

## 📚 Resources

- [SBCL Manual](http://www.sbcl.org/manual/)
- [Common Lisp Cookbook](https://lispcookbook.github.io/cl-cookbook/)
- [Practical Common Lisp](http://www.gigamonkeys.com/book/)

## 📄 License

MIT License - Learn, modify, and share freely!

---

*"The best way to learn deep learning is to implement it from scratch."*

Happy Learning! 🎉
