(in-package :mnist-ocr)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defstruct neural-network
  (w1 nil :type (simple-array single-float (* *)))  
  (b1 nil :type (simple-array single-float (*)))    
  (w2 nil :type (simple-array single-float (* *)))  
  (b2 nil :type (simple-array single-float (*))))   

(defparameter *input-size* 784)    
(defparameter *hidden-size* 128)   
(defparameter *output-size* 10)    

(defun random-normal (mean stddev)
  "Generate random number from normal distribution"
  (declare (type single-float mean stddev))
  (let ((u1 (random 1.0))
        (u2 (random 1.0)))
    (declare (type single-float u1 u2))
    (coerce (+ mean (* stddev (sqrt (* -2.0 (log u1))) (cos (* 2.0 pi u2)))) 'single-float)))

(defun xavier-init (rows cols)
  "Xavier/Glorot initialization for weight matrix"
  (declare (type fixnum rows cols))
  (let ((matrix (make-array (list rows cols) :element-type 'single-float))
        (stddev (sqrt (/ 2.0 (+ rows cols)))))
    (declare (type single-float stddev))
    (dotimes (i rows)
      (declare (type fixnum i))
      (dotimes (j cols)
        (declare (type fixnum j))
        (setf (aref matrix i j) (random-normal 0.0 stddev))))
    matrix))

(defun create-network (input-size hidden-size output-size)
  "Create a new neural network with Xavier initialization"
  (declare (type fixnum input-size hidden-size output-size))
  (make-neural-network
   :w1 (xavier-init hidden-size input-size)
   :b1 (make-array hidden-size :element-type 'single-float :initial-element 0.0)
   :w2 (xavier-init output-size hidden-size)
   :b2 (make-array output-size :element-type 'single-float :initial-element 0.0)))

(defun sigmoid (x)
  "Sigmoid activation function"
  (declare (type single-float x))
  (/ 1.0 (+ 1.0 (exp (- x)))))

(defun sigmoid-derivative (x)
  "Derivative of sigmoid function"
  (declare (type single-float x))
  (let ((s (sigmoid x)))
    (declare (type single-float s))
    (* s (- 1.0 s))))

(defun apply-activation (vector)
  "Apply sigmoid activation to vector"
  (declare (type (simple-array single-float (*)) vector))
  (let* ((size (length vector))
         (result (make-array size :element-type 'single-float)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref result i) (sigmoid (aref vector i))))
    result))

(defun apply-activation-derivative (vector)
  "Apply sigmoid derivative to vector"
  (declare (type (simple-array single-float (*)) vector))
  (let* ((size (length vector))
         (result (make-array size :element-type 'single-float)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref result i) (sigmoid-derivative (aref vector i))))
    result))

(defun matrix-multiply (matrix vector)
  "Multiply matrix by vector - returns new vector"
  (declare (type (simple-array single-float (* *)) matrix)
           (type (simple-array single-float (*)) vector))
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (result (make-array rows :element-type 'single-float)))
    (declare (type fixnum rows cols))
    (dotimes (i rows)
      (declare (type fixnum i))
      (let ((sum 0.0))
        (declare (type single-float sum))
        (dotimes (j cols)
          (declare (type fixnum j))
          (incf sum (* (aref matrix i j) (aref vector j))))
        (setf (aref result i) sum)))
    result))

(defun matrix-multiply-transpose (matrix vector)
  "Multiply transpose of matrix by vector"
  (declare (type (simple-array single-float (* *)) matrix)
           (type (simple-array single-float (*)) vector))
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (result (make-array cols :element-type 'single-float)))
    (declare (type fixnum rows cols))
    (dotimes (j cols)
      (declare (type fixnum j))
      (let ((sum 0.0))
        (declare (type single-float sum))
        (dotimes (i rows)
          (declare (type fixnum i))
          (incf sum (* (aref matrix i j) (aref vector i))))
        (setf (aref result j) sum)))
    result))

(defun vector-add (v1 v2)
  "Element-wise vector addition"
  (declare (type (simple-array single-float (*)) v1 v2))
  (let* ((size (length v1))
         (result (make-array size :element-type 'single-float)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref result i) (+ (aref v1 i) (aref v2 i))))
    result))

(defun vector-subtract (v1 v2)
  "Element-wise vector subtraction"
  (declare (type (simple-array single-float (*)) v1 v2))
  (let* ((size (length v1))
         (result (make-array size :element-type 'single-float)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref result i) (- (aref v1 i) (aref v2 i))))
    result))

(defun vector-multiply (v1 v2)
  "Element-wise vector multiplication"
  (declare (type (simple-array single-float (*)) v1 v2))
  (let* ((size (length v1))
         (result (make-array size :element-type 'single-float)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref result i) (* (aref v1 i) (aref v2 i))))
    result))

(defun softmax (vector)
  "Softmax activation function"
  (declare (type (simple-array single-float (*)) vector))
  (let* ((size (length vector))
         (max-val (reduce #'max vector))
         (exp-vec (make-array size :element-type 'single-float))
         (sum-exp 0.0))
    (declare (type fixnum size)
             (type single-float max-val sum-exp))
    (dotimes (i size)
      (declare (type fixnum i))
      (let ((exp-val (exp (- (aref vector i) max-val))))
        (declare (type single-float exp-val))
        (setf (aref exp-vec i) exp-val)
        (incf sum-exp exp-val)))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref exp-vec i) (/ (aref exp-vec i) sum-exp)))
    exp-vec))

(defun forward-pass (network input)
  "Compute network output and intermediate values"
  (declare (type neural-network network)
           (type (simple-array single-float (*)) input))
  (let* ((z1 (vector-add (matrix-multiply (neural-network-w1 network) input)
                         (neural-network-b1 network)))
         (a1 (apply-activation z1))
         (z2 (vector-add (matrix-multiply (neural-network-w2 network) a1)
                         (neural-network-b2 network)))
         (output (softmax z2)))
    (values output a1 z1)))

(defun make-one-hot (label num-classes)
  "Create one-hot encoding vector"
  (declare (type (unsigned-byte 8) label)
           (type fixnum num-classes))
  (let ((vector (make-array num-classes :element-type 'single-float :initial-element 0.0)))
    (setf (aref vector label) 1.0)
    vector))

(defun cross-entropy-loss (output target)
  "Calculate cross-entropy loss"
  (declare (type (simple-array single-float (*)) output target))
  (let ((loss 0.0))
    (declare (type single-float loss))
    (dotimes (i (length output))
      (declare (type fixnum i))
      (when (> (aref target i) 0.0)
        (decf loss (* (aref target i) (log (max 1e-7 (aref output i)))))))
    loss))

(defun argmax (vector)
  "Return index of maximum element"
  (declare (type (simple-array single-float (*)) vector))
  (let ((max-idx 0)
        (max-val (aref vector 0)))
    (declare (type fixnum max-idx)
             (type single-float max-val))
    (dotimes (i (length vector))
      (declare (type fixnum i))
      (when (> (aref vector i) max-val)
        (setf max-val (aref vector i)
              max-idx i)))
    max-idx))