(in-package :transformer)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defstruct adam-optimizer
  (learning-rate 0.001 :type single-float)
  (beta1 0.9 :type single-float)
  (beta2 0.999 :type single-float)
  (epsilon 1e-8 :type single-float)
  (t 0 :type fixnum)
  (m-weights (make-hash-table :test 'eq))
  (v-weights (make-hash-table :test 'eq)))

(defun compute-accuracy (predictions targets)
  "Compute accuracy of predictions"
  (declare (type (simple-array single-float (* * *)) predictions)
           (type (simple-array fixnum (* *)) targets))
  (let* ((batch-size (array-dimension predictions 0))
         (seq-len (array-dimension predictions 1))
         (num-classes (array-dimension predictions 2))
         (correct 0)
         (total 0))
    (dotimes (b batch-size)
      (dotimes (s seq-len)
        (let ((target (aref targets b s)))
          (when (>= target 0)
            (let ((pred-class 0)
                  (max-logit (aref predictions b s 0)))
              (loop for c from 1 below num-classes
                    do (when (> (aref predictions b s c) max-logit)
                         (setf max-logit (aref predictions b s c)
                               pred-class c)))
              (when (= pred-class target)
                (incf correct))
              (incf total))))))
    (if (zerop total) 0.0 (/ (float correct) total))))

(defun train-epoch (model x-train y-train optimizer batch-size)
  "Train model for one epoch"
  (declare (type transformer model)
           (type (simple-array fixnum (* *)) x-train y-train)
           (type adam-optimizer optimizer)
           (type fixnum batch-size))
  (let* ((num-samples (array-dimension x-train 0))
         (num-batches (ceiling num-samples batch-size))
         (total-loss 0.0)
         (total-accuracy 0.0))
    
    (dotimes (batch-idx num-batches)
      (multiple-value-bind (x-batch y-batch)
          (get-random-batch x-train y-train batch-size)
        
        (let* ((predictions (forward model x-batch t))
               (loss (cross-entropy-loss predictions y-batch))
               (accuracy (compute-accuracy predictions y-batch)))
          
          (incf total-loss loss)
          (incf total-accuracy accuracy)
          
          (when (zerop (mod batch-idx 10))
            (format t "Batch ~A/~A - Loss: ~,4F, Accuracy: ~,2F%~%"
                    batch-idx num-batches loss (* accuracy 100))))))
    
    (values (/ total-loss num-batches)
            (/ total-accuracy num-batches))))

(defun evaluate (model x-test y-test &optional (batch-size 32))
  "Evaluate model on test set"
  (declare (type transformer model)
           (type (simple-array fixnum (* *)) x-test y-test)
           (type fixnum batch-size))
  (let* ((num-samples (array-dimension x-test 0))
         (num-batches (ceiling num-samples batch-size))
         (total-loss 0.0)
         (total-accuracy 0.0)
         (all-predictions '()))
    
    (dotimes (batch-idx num-batches)
      (let* ((start-idx (* batch-idx batch-size))
             (end-idx (min (* (1+ batch-idx) batch-size) num-samples))
             (actual-batch-size (- end-idx start-idx)))
        
        (when (> actual-batch-size 0)
          (let ((x-batch (make-array (list actual-batch-size (array-dimension x-test 1))
                                     :element-type 'fixnum))
                (y-batch (make-array (list actual-batch-size (array-dimension y-test 1))
                                     :element-type 'fixnum)))
            
            (loop for i from 0 below actual-batch-size
                  for idx from start-idx
                  do (dotimes (j (array-dimension x-test 1))
                       (setf (aref x-batch i j) (aref x-test idx j))
                       (setf (aref y-batch i j) (aref y-test idx j))))
            
            (let* ((predictions (forward model x-batch nil))
                   (loss (cross-entropy-loss predictions y-batch))
                   (accuracy (compute-accuracy predictions y-batch)))
              
              (incf total-loss loss)
              (incf total-accuracy accuracy)
              (push predictions all-predictions))))))
    
    (values (/ total-accuracy num-batches)
            (nreverse all-predictions)
            (/ total-loss num-batches))))

(defun train-transformer (model x-train y-train x-test y-test 
                          &key (epochs 10) (batch-size 64) (learning-rate 0.003))
  "Train transformer model"
  (declare (type transformer model)
           (type (simple-array fixnum (* *)) x-train y-train x-test y-test)
           (type fixnum epochs batch-size)
           (type single-float learning-rate))
  
  (let ((optimizer (make-adam-optimizer :learning-rate learning-rate)))
    
    (dotimes (epoch epochs)
      (format t "~%Epoch ~A/~A~%" (1+ epoch) epochs)
      
      (multiple-value-bind (train-loss train-acc)
          (train-epoch model x-train y-train optimizer batch-size)
        (format t "Training - Loss: ~,4F, Accuracy: ~,2F%~%" 
                train-loss (* train-acc 100)))
      
      (multiple-value-bind (test-acc predictions test-loss)
          (evaluate model x-test y-test batch-size)
        (format t "Test - Loss: ~,4F, Accuracy: ~,2F%~%" 
                test-loss (* test-acc 100)))
      
      (when (< epoch (1- epochs))
        (setf (adam-optimizer-learning-rate optimizer)
              (/ (adam-optimizer-learning-rate optimizer) 1.2))
        (format t "Reducing learning rate to ~,4F~%" 
                (adam-optimizer-learning-rate optimizer))))))