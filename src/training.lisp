(in-package :mnist-ocr)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defparameter *print-training-progress* t
  "Whether to print training progress")

(defun outer-product (v1 v2)
  "Compute outer product of two vectors"
  (declare (type (simple-array single-float (*)) v1 v2))
  (let* ((rows (length v1))
         (cols (length v2))
         (result (make-array (list rows cols) :element-type 'single-float)))
    (declare (type fixnum rows cols))
    (dotimes (i rows)
      (declare (type fixnum i))
      (dotimes (j cols)
        (declare (type fixnum j))
        (setf (aref result i j) (* (aref v1 i) (aref v2 j)))))
    result))

(defun update-weights (weights gradient learning-rate)
  "Update weight matrix using gradient descent"
  (declare (type (simple-array single-float (* *)) weights gradient)
           (type single-float learning-rate))
  (let ((rows (array-dimension weights 0))
        (cols (array-dimension weights 1)))
    (declare (type fixnum rows cols))
    (dotimes (i rows)
      (declare (type fixnum i))
      (dotimes (j cols)
        (declare (type fixnum j))
        (decf (aref weights i j) (* learning-rate (aref gradient i j))))))
  weights)

(defun update-biases (biases gradient learning-rate)
  "Update bias vector using gradient descent"
  (declare (type (simple-array single-float (*)) biases gradient)
           (type single-float learning-rate))
  (let ((size (length biases)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (decf (aref biases i) (* learning-rate (aref gradient i)))))
  biases)

(defun backward-pass (network input target output a1 z1 learning-rate)
  "Update weights using gradients from backpropagation"
  (declare (type neural-network network)
           (type (simple-array single-float (*)) input target output a1 z1)
           (type single-float learning-rate))
  
  (let* ((output-error (vector-subtract output target))
         
         (w2-gradient (outer-product output-error a1))
         (b2-gradient output-error)
         
         (hidden-error (vector-multiply 
                        (matrix-multiply-transpose (neural-network-w2 network) 
                                                   output-error)
                        (apply-activation-derivative z1)))
         
         (w1-gradient (outer-product hidden-error input))
         (b1-gradient hidden-error))
    
    (update-weights (neural-network-w2 network) w2-gradient learning-rate)
    (update-weights (neural-network-w1 network) w1-gradient learning-rate)
    (update-biases (neural-network-b2 network) b2-gradient learning-rate)
    (update-biases (neural-network-b1 network) b1-gradient learning-rate)))

(defun shuffle-data (images labels)
  "Shuffle images and labels in unison"
  (declare (type (simple-array t (*)) images)
           (type (simple-array (unsigned-byte 8) (*)) labels))
  (let ((n (length images))
        (indices (make-array (length images) :element-type 'fixnum)))
    (declare (type fixnum n))
    (dotimes (i n)
      (declare (type fixnum i))
      (setf (aref indices i) i))
    
    (dotimes (i (1- n))
      (declare (type fixnum i))
      (let ((j (+ i (random (- n i)))))
        (declare (type fixnum j))
        (rotatef (aref indices i) (aref indices j))))
    
    (values
     (map 'vector (lambda (idx) (aref images idx)) indices)
     (map '(vector (unsigned-byte 8)) (lambda (idx) (aref labels idx)) indices))))

(defun train-network (network images labels epochs learning-rate)
  "Train the network on MNIST data"
  (declare (type neural-network network)
           (type (simple-array t (*)) images)
           (type (simple-array (unsigned-byte 8) (*)) labels)
           (type fixnum epochs)
           (type single-float learning-rate))
  
  (let ((num-examples (length images)))
    (declare (type fixnum num-examples))
    
    (dotimes (epoch epochs)
      (declare (type fixnum epoch))
      
      (multiple-value-bind (shuffled-images shuffled-labels)
          (shuffle-data images labels)
        
        (let ((total-loss 0.0)
              (correct 0))
          (declare (type single-float total-loss)
                   (type fixnum correct))
          
          (dotimes (i num-examples)
            (declare (type fixnum i))
            
            (let* ((input (normalize-image (aref shuffled-images i)))
                   (target (make-one-hot (aref shuffled-labels i) 10)))
              
              (multiple-value-bind (output a1 z1)
                  (forward-pass network input)
                
                (backward-pass network input target output a1 z1 learning-rate)
                
                (incf total-loss (cross-entropy-loss output target))
                
                (when (= (argmax output) (aref shuffled-labels i))
                  (incf correct))))
            
            (when (and *print-training-progress* (zerop (mod i 1000)))
              (format t "Epoch ~D: ~D/~D examples processed~%" 
                      epoch i num-examples)))
          
          (when *print-training-progress*
            (format t "Epoch ~D: Loss = ~,4F, Accuracy = ~,2F%~%" 
                    epoch 
                    (/ total-loss num-examples)
                    (* 100.0 (/ correct num-examples))))))))
  
  network)