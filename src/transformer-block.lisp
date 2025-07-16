(in-package :transformer)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defclass transformer-block ()
  ((embed-dim :initarg :embed-dim :reader embed-dim)
   (num-heads :initarg :num-heads :reader num-heads)
   (ff-dim :initarg :ff-dim :reader ff-dim)
   (dropout-rate :initarg :dropout-rate :initform 0.1 :reader dropout-rate)
   (multi-head-attn :accessor multi-head-attn)
   (ff1-weight :accessor ff1-weight)
   (ff1-bias :accessor ff1-bias)
   (ff2-weight :accessor ff2-weight)
   (ff2-bias :accessor ff2-bias)
   (ln1-gamma :accessor ln1-gamma)
   (ln1-beta :accessor ln1-beta)
   (ln2-gamma :accessor ln2-gamma)
   (ln2-beta :accessor ln2-beta)))

(defmethod initialize-instance :after ((block transformer-block) &key)
  (with-slots (embed-dim num-heads ff-dim multi-head-attn
               ff1-weight ff1-bias ff2-weight ff2-bias
               ln1-gamma ln1-beta ln2-gamma ln2-beta) block
    (setf multi-head-attn (make-instance 'multi-head-attention
                                         :embed-dim embed-dim
                                         :num-heads num-heads
                                         :head-dim (/ embed-dim num-heads)))
    (setf ff1-weight (scaled-uniform embed-dim ff-dim))
    (setf ff1-bias (make-array ff-dim :element-type 'single-float :initial-element 0.0))
    (setf ff2-weight (scaled-uniform ff-dim embed-dim))
    (setf ff2-bias (make-array embed-dim :element-type 'single-float :initial-element 0.0))
    (setf ln1-gamma (make-array embed-dim :element-type 'single-float :initial-element 1.0))
    (setf ln1-beta (make-array embed-dim :element-type 'single-float :initial-element 0.0))
    (setf ln2-gamma (make-array embed-dim :element-type 'single-float :initial-element 1.0))
    (setf ln2-beta (make-array embed-dim :element-type 'single-float :initial-element 0.0))))

(defun dropout (x rate &optional training)
  (declare (type (simple-array single-float (* *)) x)
           (type single-float rate))
  (if (and training (> rate 0.0))
      (let ((mask (make-array (array-dimensions x) :element-type 'single-float)))
        (dotimes (i (array-total-size x))
          (setf (row-major-aref mask i)
                (if (< (random 1.0) rate) 0.0 1.0)))
        (let ((result (make-array (array-dimensions x) :element-type 'single-float)))
          (dotimes (i (array-total-size x))
            (setf (row-major-aref result i)
                  (* (row-major-aref x i)
                     (row-major-aref mask i)
                     (/ 1.0 (- 1.0 rate)))))
          result))
      x))

(defun feed-forward (x ff1-weight ff1-bias ff2-weight ff2-bias)
  (declare (type (simple-array single-float (* *)) x ff1-weight ff2-weight)
           (type (simple-array single-float (*)) ff1-bias ff2-bias))
  (let* ((hidden (linear-transform x ff1-weight ff1-bias))
         (activated (make-array (array-dimensions hidden) :element-type 'single-float)))
    ;; Apply ReLU activation
    (dotimes (i (array-total-size hidden))
      (setf (row-major-aref activated i) (relu (row-major-aref hidden i))))
    (linear-transform activated ff2-weight ff2-bias)))

(defun apply-layer-norm (x gamma beta)
  (declare (type (simple-array single-float (* *)) x)
           (type (simple-array single-float (*)) gamma beta))
  (let* ((batch-size (array-dimension x 0))
         (embed-dim (array-dimension x 1))
         (result (make-array (array-dimensions x) :element-type 'single-float)))
    (dotimes (i batch-size)
      (let ((row (make-array embed-dim :element-type 'single-float)))
        (dotimes (j embed-dim)
          (setf (aref row j) (aref x i j)))
        (let ((normalized (layer-norm row gamma beta)))
          (dotimes (j embed-dim)
            (setf (aref result i j) (aref normalized j))))))
    result))

(defun add-residual (x residual)
  (declare (type (simple-array single-float (* *)) x residual))
  (let ((result (make-array (array-dimensions x) :element-type 'single-float)))
    (dotimes (i (array-total-size x))
      (setf (row-major-aref result i)
            (+ (row-major-aref x i) (row-major-aref residual i))))
    result))

(defmethod forward ((block transformer-block) x &optional mask training)
  (with-slots (multi-head-attn ff1-weight ff1-bias ff2-weight ff2-bias
               ln1-gamma ln1-beta ln2-gamma ln2-beta dropout-rate) block
    ;; Self-attention with residual connection
    (let* ((attn-output (forward multi-head-attn x mask))
           (attn-dropout (dropout attn-output dropout-rate training))
           (x-plus-attn (add-residual x attn-dropout))
           (norm1 (apply-layer-norm x-plus-attn ln1-gamma ln1-beta)))
      
      ;; Feed-forward with residual connection
      (let* ((ff-output (feed-forward norm1 ff1-weight ff1-bias ff2-weight ff2-bias))
             (ff-dropout (dropout ff-output dropout-rate training))
             (x-plus-ff (add-residual norm1 ff-dropout))
             (norm2 (apply-layer-norm x-plus-ff ln2-gamma ln2-beta)))
        norm2))))