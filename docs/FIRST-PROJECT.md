# Building MNIST OCR from Scratch in Common Lisp

Common Lisp offers a unique educational platform for implementing neural networks from scratch, combining interactive development with clear mathematical expression. This guide provides a comprehensive "101 style" approach to building an MNIST digit recognizer without external ML libraries.

## Starting with MNIST data loading

The MNIST dataset uses the IDX binary format, requiring careful handling of big-endian integers and binary data. The format consists of a magic number (4 bytes), dimensions (4 bytes each), followed by raw pixel data. Here's a beginner-friendly implementation:

```lisp
(defun read-u32-be (stream)
  "Read 32-bit unsigned integer in big-endian format"
  (+ (* (read-byte stream) #x1000000)
     (* (read-byte stream) #x10000)
     (* (read-byte stream) #x100)
     (read-byte stream)))

(defun load-mnist-images (filename)
  "Load MNIST image file - returns array of images"
  (with-open-file (stream filename 
                   :direction :input 
                   :element-type '(unsigned-byte 8))
    (let ((magic (read-u32-be stream))
          (num-images (read-u32-be stream))
          (rows (read-u32-be stream))
          (cols (read-u32-be stream)))
      (unless (= magic 2051)
        (error "Invalid magic number for image file"))
      (let ((images (make-array num-images))
            (image-size (* rows cols)))
        (dotimes (i num-images)
          (let ((image (make-array image-size 
                                   :element-type '(unsigned-byte 8))))
            (read-sequence image stream)
            (setf (aref images i) image)))
        (values images num-images rows cols)))))
```

The label loading follows a similar pattern but with a simpler structure. For SBCL optimization, declare types explicitly and use specialized arrays for better performance. The `(declare (optimize (speed 3) (safety 0)))` directive enables aggressive optimization.

## Neural network architecture from scratch

For MNIST classification, a simple feedforward network with one hidden layer provides an excellent learning platform. The architecture consists of 784 input neurons (28×28 pixels), 128 hidden neurons, and 10 output neurons (digits 0-9).

**Matrix operations form the foundation**. Since we're avoiding libraries, implement basic operations directly. Note that for SBCL compatibility, we need to ensure all float operations return single-float types:

```lisp
(defun matrix-multiply (matrix vector)
  "Multiply matrix by vector - returns new vector"
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (result (make-array rows :element-type 'single-float)))
    (dotimes (i rows)
      (setf (aref result i)
            (loop for j below cols
                  sum (* (aref matrix i j) (aref vector j))
                  single-float)))
    result))

(defun vector-add (v1 v2)
  "Element-wise vector addition"
  (map 'vector #'+ v1 v2))
```

**Activation functions bring non-linearity**. The sigmoid function works well for hidden layers in educational contexts:

```lisp
(defun sigmoid (x)
  (/ 1.0 (+ 1.0 (exp (- x)))))

(defun sigmoid-derivative (x)
  (let ((s (sigmoid x)))
    (* s (- 1.0 s))))

(defun apply-activation (vector)
  (map 'vector #'sigmoid vector))
```

For the output layer, softmax converts raw scores to probabilities:

```lisp
(defun softmax (vector)
  (let* ((max-val (reduce #'max vector))
         (exp-vec (map 'vector (lambda (x) (exp (- x max-val))) vector))
         (sum-exp (reduce #'+ exp-vec)))
    (map 'vector (lambda (x) (/ x sum-exp)) exp-vec)))
```

## Implementing backpropagation and training

Backpropagation calculates gradients through the chain rule. The implementation becomes clearer when broken into forward and backward passes:

```lisp
(defstruct neural-network
  (w1 nil :type (simple-array single-float (* *)))  ; Input to hidden weights
  (b1 nil :type (simple-array single-float (*)))    ; Hidden biases
  (w2 nil :type (simple-array single-float (* *)))  ; Hidden to output weights
  (b2 nil :type (simple-array single-float (*))))   ; Output biases

(defun forward-pass (network input)
  "Compute network output and intermediate values"
  (let* ((z1 (vector-add (matrix-multiply (neural-network-w1 network) input)
                         (neural-network-b1 network)))
         (a1 (apply-activation z1))
         (z2 (vector-add (matrix-multiply (neural-network-w2 network) a1)
                         (neural-network-b2 network)))
         (output (softmax z2)))
    (values output a1 z1)))
```

The backward pass computes gradients using the chain rule. For a cross-entropy loss with softmax output, the output layer gradient simplifies elegantly:

```lisp
(defun backward-pass (network input target output a1 z1 learning-rate)
  "Update weights using gradients"
  (let* ((batch-size 1)
         (output-error (vector-subtract output target))  ; Simplified for softmax+CE
         (hidden-error (vector-multiply 
                        (matrix-multiply-transpose (neural-network-w2 network) 
                                                   output-error)
                        (map 'vector #'sigmoid-derivative z1))))
    ;; Update weights and biases
    (update-weights (neural-network-w2 network) hidden-error output-error learning-rate)
    (update-weights (neural-network-w1 network) input hidden-error learning-rate)
    (update-biases (neural-network-b2 network) output-error learning-rate)
    (update-biases (neural-network-b1 network) hidden-error learning-rate)))
```

## Complete training implementation

The training loop brings everything together, processing batches of examples and tracking progress:

```lisp
(defun train-network (network images labels epochs learning-rate)
  "Train the network on MNIST data"
  (let ((num-examples (length images)))
    (dotimes (epoch epochs)
      (let ((total-loss 0.0)
            (correct 0))
        (dotimes (i num-examples)
          (let* ((input (normalize-image (aref images i)))
                 (target (make-one-hot (aref labels i) 10))
                 (output (multiple-value-bind (out a1 z1)
                            (forward-pass network input)
                          (backward-pass network input target out a1 z1 learning-rate)
                          out)))
            (incf total-loss (cross-entropy-loss output target))
            (when (= (argmax output) (aref labels i))
              (incf correct))))
        (format t "Epoch ~D: Loss = ~,4F, Accuracy = ~,2F%~%" 
                epoch 
                (/ total-loss num-examples)
                (* 100.0 (/ correct num-examples)))))))
```

Helper functions complete the implementation:

```lisp
(defun normalize-image (image)
  "Convert byte values to floats in [0,1]"
  (map 'vector (lambda (pixel) (/ pixel 255.0)) image))

(defun make-one-hot (label num-classes)
  "Create one-hot encoding vector"
  (let ((vector (make-array num-classes :initial-element 0.0)))
    (setf (aref vector label) 1.0)
    vector))

(defun argmax (vector)
  "Return index of maximum element"
  (position (reduce #'max vector) vector))
```

## SBCL-specific optimizations enhance performance

SBCL provides powerful optimization capabilities when properly configured. **Type declarations** significantly improve numerical performance:

```lisp
(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defun optimized-matrix-multiply (matrix vector result)
  (declare (type (simple-array single-float (* *)) matrix)
           (type (simple-array single-float (*)) vector result))
  (let ((rows (array-dimension matrix 0))
        (cols (array-dimension matrix 1)))
    (declare (type fixnum rows cols))
    (dotimes (i rows)
      (declare (type fixnum i))
      (let ((sum 0.0))
        (declare (type single-float sum))
        (dotimes (j cols)
          (declare (type fixnum j))
          (incf sum (* (aref matrix i j) (aref vector j))))
        (setf (aref result i) sum))))
  result)
```

Use specialized arrays for better memory layout and access patterns. The `simple-array` type eliminates bounds checking overhead in optimized code.

## Project structure for beginners

Organize your MNIST project with clear separation of concerns:

```
mnist-ocr/
├── mnist-ocr.asd           # ASDF system definition
├── packages.lisp           # Package definitions
├── src/
│   ├── data-loader.lisp   # MNIST loading functions
│   ├── neural-net.lisp    # Network implementation
│   ├── training.lisp      # Training loop
│   └── inference.lisp     # Prediction functions
├── examples/
│   └── train-mnist.lisp   # Complete example
└── data/                   # MNIST files go here
```

The ASDF system definition keeps dependencies minimal:

```lisp
(asdf:defsystem "mnist-ocr"
  :description "MNIST OCR from scratch in Common Lisp"
  :author "Your Name"
  :license "MIT"
  :serial t
  :components ((:file "packages")
               (:file "src/data-loader")
               (:file "src/neural-net")
               (:file "src/training")
               (:file "src/inference")
               (:file "examples/train-mnist")))
```

## Inference implementation for digit recognition

Once trained, the network performs inference through a simple forward pass:

```lisp
(defun predict-digit (network image)
  "Predict digit class for a single image"
  (let* ((input (normalize-image image))
         (output (forward-pass network input)))
    (argmax output)))

(defun evaluate-accuracy (network test-images test-labels)
  "Calculate accuracy on test set"
  (let ((correct 0)
        (total (length test-images)))
    (dotimes (i total)
      (when (= (predict-digit network (aref test-images i))
               (aref test-labels i))
        (incf correct)))
    (format t "Test Accuracy: ~,2F%~%" (* 100.0 (/ correct total)))))
```

## Learning resources and next steps

Common Lisp's interactive development environment makes it ideal for learning machine learning concepts. The REPL allows immediate experimentation with network components, making debugging intuitive. Key educational advantages include the ability to inspect intermediate values during training, modify running code, and experiment with different architectures interactively.

## Implementation Notes

When implementing the random normal distribution for weight initialization, ensure type compatibility with SBCL:

```lisp
(defun random-normal (mean stddev)
  "Generate random number from normal distribution"
  (declare (type single-float mean stddev))
  (let ((u1 (random 1.0))
        (u2 (random 1.0)))
    (declare (type single-float u1 u2))
    (coerce (+ mean (* stddev (sqrt (* -2.0 (log u1))) (cos (* 2.0 pi u2)))) 'single-float)))
```

The `coerce` function ensures the result is single-float, preventing type mismatch errors during compilation.

For further learning, explore weight initialization strategies like Xavier initialization, implement momentum in gradient descent, or add dropout regularization. The **simple-neural-network** library provides a clean reference implementation, while **MGL** demonstrates production-quality approaches. The symbolic nature of Lisp also enables interesting experiments like the pure-lambda neural networks demonstrated in Woodrush's blog, showing how neural computation emerges from basic list operations.

This foundation provides everything needed to understand neural networks at a fundamental level while leveraging Common Lisp's unique strengths for educational machine learning projects.
