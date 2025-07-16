#!/usr/bin/sbcl --script

(require :asdf)
(push #P"./" asdf:*central-registry*)
(asdf:load-system :mnist-ocr)
(in-package :mnist-ocr)

(format t "~%MNIST OCR Demo~%")
(format t "==============~%~%")

;; Train a small network quickly for demo
(handler-case
    (multiple-value-bind (train-images train-labels test-images test-labels)
        (load-mnist-data "data/")
      
      (format t "Creating and training network...~%")
      (let ((network (create-network *input-size* 64 *output-size*)))
        
        ;; Quick training on subset
        (setf *print-training-progress* nil)
        (train-network network 
                       (subseq train-images 0 5000)
                       (subseq train-labels 0 5000)
                       3 0.1)
        (setf *print-training-progress* t)
        
        ;; Test on a few examples
        (format t "~%Testing on sample images:~%")
        (format t "========================~%")
        
        (dotimes (i 5)
          (let* ((idx (random 1000))
                 (image (aref test-images idx))
                 (actual (aref test-labels idx))
                 (predicted (predict-digit network image)))
            
            (format t "~%Test image #~D:~%" idx)
            (format t "Actual digit: ~D~%" actual)
            (format t "Predicted: ~D " predicted)
            (if (= actual predicted)
                (format t "✓ CORRECT~%")
                (format t "✗ INCORRECT~%"))
            
            ;; Show the image
            (display-image image)))
        
        ;; Overall accuracy
        (format t "~%Calculating accuracy on 1000 test samples...~%")
        (let ((correct 0))
          (dotimes (i 1000)
            (when (= (predict-digit network (aref test-images i))
                     (aref test-labels i))
              (incf correct)))
          (format t "Accuracy: ~,1F% (~D/1000)~%~%" 
                  (* 100.0 (/ correct 1000))
                  correct))))
    
    (error (e)
      (format t "Error: ~A~%" e)))