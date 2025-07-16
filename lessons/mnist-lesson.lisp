;;;; MNIST Neural Network from Scratch
;;;; A complete implementation for learning handwritten digit recognition

(defpackage :mnist-lesson
  (:use :cl))

(in-package :mnist-lesson)

;; Optimize for speed
(declaim (optimize (speed 3) (safety 0) (debug 0)))

;;; ========================================
;;; MNIST Data Loader
;;; ========================================

(defun read-int32 (stream)
  "Read a 32-bit big-endian integer"
  (let ((bytes (make-array 4 :element-type '(unsigned-byte 8))))
    (read-sequence bytes stream)
    (+ (ash (aref bytes 0) 24)
       (ash (aref bytes 1) 16)
       (ash (aref bytes 2) 8)
       (aref bytes 3))))

(defun load-mnist-images (filename)
  "Load MNIST image file"
  (with-open-file (stream filename :element-type '(unsigned-byte 8))
    (let* ((magic (read-int32 stream))
           (num-images (read-int32 stream))
           (rows (read-int32 stream))
           (cols (read-int32 stream))
           (image-size (* rows cols)))
      
      (assert (= magic 2051) () "Invalid magic number for images: ~A" magic)
      (format t "Loading ~A images (~Ax~A pixels)...~%" num-images rows cols)
      
      ;; Read all images
      (let ((images (make-array num-images)))
        (dotimes (i num-images)
          (let ((image (make-array image-size :element-type '(unsigned-byte 8))))
            (read-sequence image stream)
            (setf (aref images i) image)))
        images))))

(defun load-mnist-labels (filename)
  "Load MNIST label file"
  (with-open-file (stream filename :element-type '(unsigned-byte 8))
    (let* ((magic (read-int32 stream))
           (num-labels (read-int32 stream)))
      
      (assert (= magic 2049) () "Invalid magic number for labels: ~A" magic)
      (format t "Loading ~A labels...~%" num-labels)
      
      ;; Read all labels
      (let ((labels (make-array num-labels :element-type '(unsigned-byte 8))))
        (read-sequence labels stream)
        labels))))

(defun normalize-image (image)
  "Convert byte image to normalized floats"
  (map '(simple-array single-float (*))
       (lambda (pixel) (/ (float pixel) 255.0))
       image))

(defun one-hot-encode (label)
  "Convert label to one-hot vector"
  (let ((vec (make-array 10 :element-type 'single-float :initial-element 0.0)))
    (setf (aref vec label) 1.0)
    vec))

;;; ========================================
;;; Neural Network Structure
;;; ========================================

(defstruct layer
  weights    ; Weight matrix
  biases     ; Bias vector
  activation ; Activation function name
  output     ; Cached output for backprop
  input)     ; Cached input for backprop

(defstruct network
  layers)    ; List of layers

(defun random-normal (mean stddev)
  "Generate random number from normal distribution (Box-Muller)"
  (let* ((u1 (max 1e-7 (random 1.0)))
         (u2 (random 1.0))
         (z0 (sqrt (* -2 (log u1))))
         (z1 (* z0 (cos (* 2 pi u2)))))
    (+ mean (* stddev z1))))

(defun xavier-init (fan-in fan-out)
  "Xavier/Glorot initialization"
  (let* ((scale (sqrt (/ 2.0 (+ fan-in fan-out))))
         (weights (make-array (list fan-in fan-out) :element-type 'single-float)))
    (dotimes (i fan-in)
      (dotimes (j fan-out)
        (setf (aref weights i j) (random-normal 0.0 scale))))
    weights))

(defun create-layer (input-size output-size activation)
  "Create a neural network layer"
  (make-layer
   :weights (xavier-init input-size output-size)
   :biases (make-array output-size :element-type 'single-float :initial-element 0.0)
   :activation activation))

(defun create-network (&rest layer-sizes)
  "Create a neural network with specified layer sizes"
  (let ((layers '()))
    (loop for i from 0 below (1- (length layer-sizes))
          for input-size = (nth i layer-sizes)
          for output-size = (nth (1+ i) layer-sizes)
          for activation = (if (= i (- (length layer-sizes) 2)) 'softmax 'sigmoid)
          do (push (create-layer input-size output-size activation) layers))
    (make-network :layers (nreverse layers))))

;;; ========================================
;;; Activation Functions
;;; ========================================

(defun sigmoid (x)
  "Sigmoid activation"
  (cond
    ((> x 50.0) 1.0)         ; prevent overflow
    ((< x -50.0) 0.0)        ; prevent underflow
    (t (/ 1.0 (+ 1.0 (exp (- x)))))))

(defun sigmoid-derivative (x)
  "Derivative of sigmoid"
  (* x (- 1.0 x)))

(defun softmax (x)
  "Softmax activation for output layer"
  (let* ((max-val (reduce #'max x))
         (exp-vals (map '(simple-array single-float (*))
                        (lambda (xi) 
                          (let ((val (- xi max-val)))
                            (if (< val -50.0)  ; prevent underflow
                                0.0
                                (exp val))))
                        x))
         (sum-exp (max 1e-7 (reduce #'+ exp-vals))))  ; prevent division by zero
    (map '(simple-array single-float (*))
         (lambda (ei) (/ ei sum-exp))
         exp-vals)))

;;; ========================================
;;; Forward Propagation
;;; ========================================

(defun matrix-vector-multiply (matrix vector)
  "Multiply matrix by vector"
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (result (make-array rows :element-type 'single-float :initial-element 0.0)))
    (dotimes (i rows)
      (dotimes (j cols)
        (incf (aref result i) (* (aref matrix i j) (aref vector j)))))
    result))

(defun vector-add (a b)
  "Add two vectors"
  (map '(simple-array single-float (*)) #'+ a b))

(defun apply-activation (x activation)
  "Apply activation function"
  (ecase activation
    (sigmoid (map '(simple-array single-float (*)) #'sigmoid x))
    (softmax (softmax x))
    (linear x)))

(defun forward-pass (network input)
  "Forward propagation through network"
  (let ((current input))
    (dolist (layer (network-layers network))
      ;; Cache input for backprop
      (setf (layer-input layer) current)
      
      ;; Linear transformation: Wx + b
      (let* ((z (vector-add (matrix-vector-multiply (layer-weights layer) current)
                            (layer-biases layer)))
             (output (apply-activation z (layer-activation layer))))
        
        ;; Cache output for backprop
        (setf (layer-output layer) output)
        (setf current output)))
    current))

;;; ========================================
;;; Backpropagation
;;; ========================================

(defun cross-entropy-derivative (predicted actual)
  "Derivative of cross-entropy loss with softmax"
  (map '(simple-array single-float (*)) #'- predicted actual))

(defun outer-product (a b)
  "Compute outer product of two vectors"
  (let* ((m (length a))
         (n (length b))
         (result (make-array (list m n) :element-type 'single-float)))
    (dotimes (i m)
      (dotimes (j n)
        (setf (aref result i j) (* (aref a i) (aref b j)))))
    result))

(defun matrix-transpose-vector-multiply (matrix vector)
  "Multiply transpose of matrix by vector"
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (result (make-array cols :element-type 'single-float :initial-element 0.0)))
    (dotimes (i rows)
      (dotimes (j cols)
        (incf (aref result j) (* (aref matrix i j) (aref vector i)))))
    result))

(defun backward-pass (network input target learning-rate)
  "Backpropagation with weight updates"
  ;; First, do forward pass to cache activations
  (let ((output (forward-pass network input)))
    
    ;; Compute initial error at output layer
    (let ((layers (network-layers network))
          (errors (cross-entropy-derivative output target)))
      
      ;; Backpropagate through layers
      (loop for i from (1- (length layers)) downto 0
            for layer = (nth i layers)
            do (let* ((input-to-layer (if (> i 0)
                                          (layer-output (nth (1- i) layers))
                                          input))
                      ;; Compute gradients
                      (weight-grad (outer-product errors input-to-layer))
                      (bias-grad errors))
                 
                 ;; Update weights and biases with gradient clipping
                 (dotimes (r (array-dimension (layer-weights layer) 0))
                   (dotimes (c (array-dimension (layer-weights layer) 1))
                     (let ((grad (aref weight-grad r c)))
                       (when (and (numberp grad) (not (sb-ext:float-nan-p grad)))
                         (setf grad (min 1.0 (max -1.0 grad)))
                         (decf (aref (layer-weights layer) r c)
                               (* learning-rate grad))))))
                 
                 (dotimes (j (length (layer-biases layer)))
                   (let ((grad (aref bias-grad j)))
                     (when (and (numberp grad) (not (sb-ext:float-nan-p grad)))
                       (setf grad (min 1.0 (max -1.0 grad)))
                       (decf (aref (layer-biases layer) j)
                             (* learning-rate grad)))))
                 
                 ;; Propagate error to previous layer
                 (when (> i 0)
                   (let ((prev-errors (matrix-transpose-vector-multiply 
                                      (layer-weights layer) errors)))
                     ;; Apply derivative of activation
                     (when (eq (layer-activation (nth (1- i) layers)) 'sigmoid)
                       (let ((prev-output (layer-output (nth (1- i) layers))))
                         (setf prev-errors
                               (map '(simple-array single-float (*))
                                    (lambda (e o) (* e (sigmoid-derivative o)))
                                    prev-errors prev-output))))
                     (setf errors prev-errors))))))
    
    ;; Return loss for monitoring
    (- (reduce #'+ (map 'list 
                        (lambda (tgt o) (if (> tgt 0.5) (log (max 1e-7 o)) 0))
                        target output)))))

;;; ========================================
;;; Training Functions
;;; ========================================

(defun train-epoch (network images labels learning-rate)
  "Train network for one epoch"
  (let ((total-loss 0.0)
        (correct 0)
        (n (length images)))
    
    ;; Shuffle indices
    (let ((indices (loop for i below n collect i)))
      (loop for i from (1- n) downto 1
            do (let ((j (random (1+ i))))
                 (rotatef (nth i indices) (nth j indices))))
      
      ;; Train on each example
      (loop for idx in indices
            for i from 0
            do (let* ((image (normalize-image (aref images idx)))
                      (label (one-hot-encode (aref labels idx)))
                      (loss (backward-pass network image label learning-rate))
                      (output (forward-pass network image))
                      (predicted (position (reduce #'max output) output)))
                 
                 (incf total-loss loss)
                 (when (= predicted (aref labels idx))
                   (incf correct))
                 
                 ;; Progress update
                 (when (zerop (mod i 1000))
                   (format t "  Processed ~A/~A images...~%" i n)))))
    
    (values (/ total-loss n) (/ correct (float n)))))

(defun evaluate-network (network images labels)
  "Evaluate network accuracy"
  (let ((correct 0))
    (dotimes (i (length images))
      (let* ((image (normalize-image (aref images i)))
             (output (forward-pass network image))
             (predicted (position (reduce #'max output) output)))
        (when (= predicted (aref labels i))
          (incf correct))))
    (/ correct (float (length images)))))

;;; ========================================
;;; Visualization Helpers
;;; ========================================

(defun print-digit (image)
  "Print ASCII representation of digit"
  (dotimes (row 28)
    (dotimes (col 28)
      (let ((pixel (aref image (+ (* row 28) col))))
        (princ (cond ((< pixel 50) " ")
                     ((< pixel 150) ".")
                     ((< pixel 200) "*")
                     (t "#")))))
    (terpri)))

(defun test-predictions (network images labels n)
  "Show some test predictions"
  (format t "~%Sample predictions:~%")
  (dotimes (i (min n (length images)))
    (let* ((image (aref images i))
           (norm-image (normalize-image image))
           (output (forward-pass network norm-image))
           (predicted (position (reduce #'max output) output))
           (actual (aref labels i)))
      
      (format t "~%Example ~A - Actual: ~A, Predicted: ~A ~A~%"
              i actual predicted
              (if (= actual predicted) "✓" "✗"))
      
      (print-digit image)
      
      ;; Show confidence scores
      (format t "Confidence scores: ")
      (dotimes (j 10)
        (format t "~A:~,2F " j (aref output j)))
      (terpri))))

;;; ========================================
;;; Main Training Program
;;; ========================================

(defun run-mnist-demo ()
  "Run the complete MNIST training demo"
  (format t "~%=== MNIST Neural Network Demo ===~%")
  (format t "Training a neural network to recognize handwritten digits~%~%")
  
  ;; Load data
  (format t "Loading MNIST dataset...~%")
  (let ((train-images (load-mnist-images "data/train-images-idx3-ubyte"))
        (train-labels (load-mnist-labels "data/train-labels-idx1-ubyte"))
        (test-images (load-mnist-images "data/t10k-images-idx3-ubyte"))
        (test-labels (load-mnist-labels "data/t10k-labels-idx1-ubyte")))
    
    ;; Create network: 784 inputs -> 128 hidden -> 10 outputs
    (format t "~%Creating neural network (784 -> 128 -> 10)...~%")
    (let ((network (create-network 784 128 10))
          (learning-rate 0.01)
          (epochs 5))
      
      ;; Use subset for faster demo
      (let ((train-subset-size 10000)
            (test-subset-size 1000))
        
        (format t "Using ~A training and ~A test examples~%~%"
                train-subset-size test-subset-size)
        
        ;; Training loop
        (format t "Starting training...~%")
        (dotimes (epoch epochs)
          (format t "~%Epoch ~A/~A~%" (1+ epoch) epochs)
          
          (multiple-value-bind (loss accuracy)
              (train-epoch network
                          (subseq train-images 0 train-subset-size)
                          (subseq train-labels 0 train-subset-size)
                          learning-rate)
            
            (format t "  Training Loss: ~,4F~%" loss)
            (format t "  Training Accuracy: ~,1F%~%" (* 100 accuracy)))
          
          ;; Test accuracy
          (let ((test-acc (evaluate-network network
                                           (subseq test-images 0 test-subset-size)
                                           (subseq test-labels 0 test-subset-size))))
            (format t "  Test Accuracy: ~,1F%~%" (* 100 test-acc)))
          
          ;; Decay learning rate
          (setf learning-rate (* learning-rate 0.9))))
      
      ;; Show some predictions
      (test-predictions network test-images test-labels 5)
      
      ;; Final evaluation
      (format t "~%Final test accuracy on full test set: ~,1F%~%"
              (* 100 (evaluate-network network test-images test-labels))))))

;;; ========================================
;;; Utility Functions
;;; ========================================

(defun save-network (network filename)
  "Save trained network to file"
  (with-open-file (stream filename :direction :output :if-exists :supersede)
    (print (list :network
                 :layers (mapcar (lambda (layer)
                                  (list :weights (layer-weights layer)
                                        :biases (layer-biases layer)
                                        :activation (layer-activation layer)))
                                (network-layers network)))
           stream))
  (format t "Network saved to ~A~%" filename))

(defun load-network (filename)
  "Load network from file"
  (with-open-file (stream filename)
    (let* ((data (read stream))
           (layers (mapcar (lambda (layer-data)
                            (make-layer :weights (getf layer-data :weights)
                                       :biases (getf layer-data :biases)
                                       :activation (getf layer-data :activation)))
                          (getf data :layers))))
      (make-network :layers layers))))

;;; ========================================
;;; Interactive Functions
;;; ========================================

(defun quick-test ()
  "Quick test with small dataset"
  (format t "~%Quick test with 1000 training examples...~%")
  (let ((train-images (load-mnist-images "data/train-images-idx3-ubyte"))
        (train-labels (load-mnist-labels "data/train-labels-idx1-ubyte"))
        (test-images (load-mnist-images "data/t10k-images-idx3-ubyte"))
        (test-labels (load-mnist-labels "data/t10k-labels-idx1-ubyte"))
        (network (create-network 784 64 10))) ; Smaller network for quick test
    
    ;; Train on subset
    (train-epoch network
                 (subseq train-images 0 1000)
                 (subseq train-labels 0 1000)
                 0.5)
    
    ;; Test
    (let ((acc (evaluate-network network
                                 (subseq test-images 0 100)
                                 (subseq test-labels 0 100))))
      (format t "Test accuracy after 1 epoch: ~,1F%~%" (* 100 acc)))))

;;; ========================================
;;; Run the demo
;;; ========================================

(run-mnist-demo)
