(in-package :transformer)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defclass transformer ()
  ((num-symbols :initarg :num-symbols :reader num-symbols)
   (max-len :initarg :max-len :reader max-len)
   (num-layers :initarg :num-layers :reader num-layers)
   (embed-dim :initarg :embed-dim :reader embed-dim)
   (num-heads :initarg :num-heads :reader num-heads)
   (ff-dim :initarg :ff-dim :reader ff-dim)
   (embedding :accessor embedding)
   (transformer-blocks :accessor transformer-blocks)
   (final-weight :accessor final-weight)))

(defmethod initialize-instance :after ((model transformer) &key)
  (with-slots (num-symbols max-len num-layers embed-dim num-heads ff-dim
               embedding transformer-blocks final-weight) model
    ;; Initialize embeddings for both position and symbol
    (setf embedding (scaled-uniform (+ max-len num-symbols) embed-dim))
    
    ;; Initialize transformer blocks
    (setf transformer-blocks
          (loop for i from 0 below num-layers
                collect (make-instance 'transformer-block
                                       :embed-dim embed-dim
                                       :num-heads num-heads
                                       :ff-dim ff-dim)))
    
    ;; Initialize final projection layer
    (setf final-weight (scaled-uniform embed-dim num-symbols))))

(defun one-hot (indices num-classes)
  (declare (type (simple-array fixnum (*)) indices)
           (type fixnum num-classes))
  (let* ((n (length indices))
         (result (make-array (list n num-classes) :element-type 'single-float :initial-element 0.0)))
    (dotimes (i n)
      (setf (aref result i (aref indices i)) 1.0))
    result))

(defun create-position-encoding (batch-size seq-len)
  (let ((result (make-array (list batch-size seq-len seq-len)
                            :element-type 'single-float :initial-element 0.0)))
    (dotimes (b batch-size)
      (dotimes (i seq-len)
        (setf (aref result b i i) 1.0)))
    result))

(defmethod forward ((model transformer) x &optional training)
  (declare (type (simple-array fixnum (* *)) x))
  (with-slots (num-symbols max-len embed-dim embedding transformer-blocks final-weight) model
    (let* ((batch-size (array-dimension x 0))
           (seq-len (array-dimension x 1)))
      
      ;; Create position encodings
      (let* ((pos-encoding (create-position-encoding batch-size seq-len))
             (pos-flat (make-array (list (* batch-size seq-len) seq-len)
                                   :element-type 'single-float
                                   :displaced-to pos-encoding))
             
             ;; Create symbol one-hot encodings
             (x-flat (make-array (* batch-size seq-len) :element-type 'fixnum
                                 :displaced-to x))
             (symbol-onehot (one-hot x-flat num-symbols))
             
             ;; Concatenate position and symbol encodings
             (combined (make-array (list (* batch-size seq-len) (+ seq-len num-symbols))
                                   :element-type 'single-float)))
        
        ;; Fill combined array
        (dotimes (i (* batch-size seq-len))
          (dotimes (j seq-len)
            (setf (aref combined i j) (aref pos-flat i j)))
          (dotimes (j num-symbols)
            (setf (aref combined i (+ seq-len j)) (aref symbol-onehot i j))))
        
        ;; Embed the combined input
        (let* ((embedded (matrix-multiply combined embedding))
               (x-embedded (make-array (list batch-size seq-len embed-dim)
                                       :element-type 'single-float)))
          
          ;; Reshape embedded back to 3D
          (dotimes (b batch-size)
            (dotimes (s seq-len)
              (dotimes (e embed-dim)
                (setf (aref x-embedded b s e)
                      (aref embedded (+ (* b seq-len) s) e)))))
          
          ;; Pass through transformer blocks
          (let ((current x-embedded))
            (dolist (block transformer-blocks)
              (setf current (forward block current nil training)))
            
            ;; Final projection and softmax
            (let* ((current-2d (make-array (list (* batch-size seq-len) embed-dim)
                                           :element-type 'single-float
                                           :displaced-to current))
                   (logits (matrix-multiply current-2d final-weight))
                   (output (make-array (list batch-size seq-len num-symbols)
                                       :element-type 'single-float)))
              
              ;; Apply log-softmax to each position
              (dotimes (i (* batch-size seq-len))
                (let ((logit-row (make-array num-symbols :element-type 'single-float)))
                  (dotimes (j num-symbols)
                    (setf (aref logit-row j) (aref logits i j)))
                  (let ((softmax-row (softmax logit-row)))
                    (dotimes (j num-symbols)
                      (let ((b (floor i seq-len))
                            (s (mod i seq-len)))
                        (setf (aref output b s j)
                              (log (max 1e-10 (aref softmax-row j)))))))))
              
              output)))))))

(defun cross-entropy-loss (predictions targets)
  (declare (type (simple-array single-float (* * *)) predictions)
           (type (simple-array fixnum (* *)) targets))
  (let* ((batch-size (array-dimension predictions 0))
         (seq-len (array-dimension predictions 1))
         (total-loss 0.0))
    (declare (type single-float total-loss))
    (dotimes (b batch-size)
      (dotimes (s seq-len)
        (let ((target-idx (aref targets b s)))
          (when (>= target-idx 0)  ; Ignore padding tokens
            (decf total-loss (aref predictions b s target-idx))))))
    (/ total-loss (* batch-size seq-len))))