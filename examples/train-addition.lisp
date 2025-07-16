(in-package :transformer)

(defun run-addition-experiment ()
  "Train a transformer to learn 2-digit addition"
  (format t "~%=== Transformer Addition Experiment ===~%")
  (format t "Learning to add 2-digit numbers~%~%")
  
  ;; Create model
  (let ((model (make-instance 'transformer
                              :num-symbols 10      ; digits 0-9
                              :max-len 6          ; sequence length
                              :num-layers 2       ; transformer blocks
                              :embed-dim 128      ; embedding dimension
                              :num-heads 4        ; attention heads
                              :ff-dim 32)))       ; feed-forward dimension
    
    ;; Generate dataset
    (format t "Generating addition dataset...~%")
    (multiple-value-bind (x-train y-train x-test y-test)
        (make-addition-dataset)
      
      (format t "Training set: ~A examples~%" (array-dimension x-train 0))
      (format t "Test set: ~A examples~%" (array-dimension x-test 0))
      (format t "~%Starting training...~%")
      
      ;; Train model
      (train-transformer model x-train y-train x-test y-test
                         :epochs 10
                         :batch-size 64
                         :learning-rate 0.003)
      
      ;; Test with some examples
      (format t "~%Testing on some examples:~%")
      (dotimes (i 10)
        (let* ((idx (random (array-dimension x-test 0)))
               (x-example (make-array '(1 6) :element-type 'fixnum))
               (y-true (make-array 6 :element-type 'fixnum)))
          
          ;; Copy example
          (dotimes (j 6)
            (setf (aref x-example 0 j) (aref x-test idx j))
            (setf (aref y-true j) (aref y-test idx j)))
          
          ;; Get prediction
          (let* ((pred (forward model x-example nil))
                 (pred-digits (make-array 6 :element-type 'fixnum)))
            
            ;; Extract predicted digits
            (dotimes (j 6)
              (let ((max-idx 0)
                    (max-val (aref pred 0 j 0)))
                (loop for k from 1 below 10
                      when (> (aref pred 0 j k) max-val)
                      do (setf max-idx k
                               max-val (aref pred 0 j k)))
                (setf (aref pred-digits j) max-idx)))
            
            ;; Display result
            (format t "~A~A + ~A~A = ~A~A~A (predicted: ~A~A~A)~%"
                    (aref x-example 0 0) (aref x-example 0 1)
                    (aref x-example 0 2) (aref x-example 0 3)
                    (aref y-true 4) (aref y-true 5) (aref y-true 6)
                    (aref pred-digits 4) (aref pred-digits 5) (aref pred-digits 6))))))))

;; Run the experiment when loaded
(defun main ()
  (run-addition-experiment))

;; To run:
;; (asdf:load-system :transformer)
;; (in-package :transformer)
;; (main)