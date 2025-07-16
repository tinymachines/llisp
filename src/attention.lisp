(in-package :transformer)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defun scaled-dot-product-attention (query key value &optional mask)
  (declare (type (simple-array single-float (* *)) query key value))
  (let* ((d-k (array-dimension query 1))
         (scores (matrix-multiply query (transpose key)))
         (batch-size (array-dimension scores 0))
         (seq-len (array-dimension scores 1)))
    
    ;; Scale scores
    (dotimes (i batch-size)
      (dotimes (j seq-len)
        (setf (aref scores i j) (/ (aref scores i j) (sqrt (float d-k))))))
    
    ;; Apply mask if provided
    (when mask
      (dotimes (i batch-size)
        (dotimes (j seq-len)
          (when (zerop (aref mask i j))
            (setf (aref scores i j) -1e9)))))
    
    ;; Apply softmax to each row
    (let ((attention-weights (make-array (array-dimensions scores) :element-type 'single-float)))
      (dotimes (i batch-size)
        (let ((row (make-array seq-len :element-type 'single-float)))
          (dotimes (j seq-len)
            (setf (aref row j) (aref scores i j)))
          (let ((softmax-row (softmax row)))
            (dotimes (j seq-len)
              (setf (aref attention-weights i j) (aref softmax-row j))))))
      
      ;; Apply attention weights to values
      (matrix-multiply attention-weights value))))

(defclass multi-head-attention ()
  ((num-heads :initarg :num-heads :reader num-heads)
   (head-dim :initarg :head-dim :reader head-dim)
   (embed-dim :initarg :embed-dim :reader embed-dim)
   (w-query :accessor w-query)
   (w-key :accessor w-key)
   (w-value :accessor w-value)
   (w-out :accessor w-out)
   (b-query :accessor b-query)
   (b-key :accessor b-key)
   (b-value :accessor b-value)
   (b-out :accessor b-out)))

(defmethod initialize-instance :after ((mha multi-head-attention) &key)
  (with-slots (embed-dim num-heads head-dim w-query w-key w-value w-out
               b-query b-key b-value b-out) mha
    (setf w-query (scaled-uniform embed-dim embed-dim))
    (setf w-key (scaled-uniform embed-dim embed-dim))
    (setf w-value (scaled-uniform embed-dim embed-dim))
    (setf w-out (scaled-uniform embed-dim embed-dim))
    (setf b-query (make-array embed-dim :element-type 'single-float :initial-element 0.0))
    (setf b-key (make-array embed-dim :element-type 'single-float :initial-element 0.0))
    (setf b-value (make-array embed-dim :element-type 'single-float :initial-element 0.0))
    (setf b-out (make-array embed-dim :element-type 'single-float :initial-element 0.0))))

(defun linear-transform (input weight bias)
  (declare (type (simple-array single-float (* *)) input weight)
           (type (simple-array single-float (*)) bias))
  (let* ((batch-size (array-dimension input 0))
         (input-dim (array-dimension input 1))
         (output-dim (array-dimension weight 1))
         (result (make-array (list batch-size output-dim) :element-type 'single-float)))
    (dotimes (i batch-size)
      (dotimes (j output-dim)
        (let ((sum (aref bias j)))
          (dotimes (k input-dim)
            (incf sum (* (aref input i k) (aref weight k j))))
          (setf (aref result i j) sum))))
    result))

(defmethod forward ((mha multi-head-attention) x &optional mask)
  (with-slots (num-heads head-dim embed-dim w-query w-key w-value w-out
               b-query b-key b-value b-out) mha
    (let* ((batch-size (array-dimension x 0))
           (seq-len (if (= (array-rank x) 3)
                        (array-dimension x 1)
                        1))
           (x-2d (if (= (array-rank x) 3)
                     (make-array (list (* batch-size seq-len) embed-dim)
                                 :element-type 'single-float
                                 :displaced-to x)
                     x)))
      
      ;; Linear projections
      (let* ((q (linear-transform x-2d w-query b-query))
             (k (linear-transform x-2d w-key b-key))
             (v (linear-transform x-2d w-value b-value)))
        
        ;; Reshape for multi-head attention
        (let* ((q-heads (reshape-for-heads q batch-size seq-len num-heads head-dim))
               (k-heads (reshape-for-heads k batch-size seq-len num-heads head-dim))
               (v-heads (reshape-for-heads v batch-size seq-len num-heads head-dim))
               (attention-output (make-array (list batch-size num-heads seq-len head-dim)
                                             :element-type 'single-float)))
          
          ;; Apply attention for each head
          (dotimes (h num-heads)
            (let ((q-h (extract-head q-heads h))
                  (k-h (extract-head k-heads h))
                  (v-h (extract-head v-heads h)))
              (let ((head-output (scaled-dot-product-attention q-h k-h v-h mask)))
                (set-head attention-output h head-output))))
          
          ;; Concatenate heads and project
          (let* ((concat-output (concatenate-heads attention-output batch-size seq-len embed-dim))
                 (output (linear-transform concat-output w-out b-out)))
            output))))))

(defun reshape-for-heads (x batch-size seq-len num-heads head-dim)
  (let ((result (make-array (list batch-size num-heads seq-len head-dim)
                            :element-type 'single-float)))
    (dotimes (b batch-size)
      (dotimes (s seq-len)
        (dotimes (h num-heads)
          (dotimes (d head-dim)
            (setf (aref result b h s d)
                  (aref x (+ (* b seq-len) s) (+ (* h head-dim) d)))))))
    result))

(defun extract-head (x head-idx)
  (let* ((batch-size (array-dimension x 0))
         (seq-len (array-dimension x 2))
         (head-dim (array-dimension x 3))
         (result (make-array (list batch-size seq-len head-dim)
                             :element-type 'single-float)))
    (dotimes (b batch-size)
      (dotimes (s seq-len)
        (dotimes (d head-dim)
          (setf (aref result b s d)
                (aref x b head-idx s d)))))
    result))

(defun set-head (output head-idx head-data)
  (let* ((batch-size (array-dimension head-data 0))
         (seq-len (array-dimension head-data 1))
         (head-dim (array-dimension head-data 2)))
    (dotimes (b batch-size)
      (dotimes (s seq-len)
        (dotimes (d head-dim)
          (setf (aref output b head-idx s d)
                (aref head-data b s d)))))))

(defun concatenate-heads (x batch-size seq-len embed-dim)
  (let ((result (make-array (list (* batch-size seq-len) embed-dim)
                            :element-type 'single-float)))
    (dotimes (b batch-size)
      (dotimes (s seq-len)
        (dotimes (e embed-dim)
          (let ((h (floor e (/ embed-dim (array-dimension x 1))))
                (d (mod e (/ embed-dim (array-dimension x 1)))))
            (setf (aref result (+ (* b seq-len) s) e)
                  (aref x b h s d))))))
    result))