(in-package :mnist-ocr)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defun predict-digit (network image)
  "Predict digit class for a single image"
  (declare (type neural-network network)
           (type (simple-array (unsigned-byte 8) (*)) image))
  (let* ((input (normalize-image image))
         (output (forward-pass network input)))
    (argmax output)))

(defun predict-batch (network images)
  "Predict digits for a batch of images"
  (declare (type neural-network network)
           (type (simple-array t (*)) images))
  (let* ((batch-size (length images))
         (predictions (make-array batch-size :element-type 'fixnum)))
    (declare (type fixnum batch-size))
    (dotimes (i batch-size)
      (declare (type fixnum i))
      (setf (aref predictions i) (predict-digit network (aref images i))))
    predictions))

(defun evaluate-accuracy (network test-images test-labels)
  "Calculate accuracy on test set"
  (declare (type neural-network network)
           (type (simple-array t (*)) test-images)
           (type (simple-array (unsigned-byte 8) (*)) test-labels))
  (let ((correct 0)
        (total (length test-images)))
    (declare (type fixnum correct total))
    (dotimes (i total)
      (declare (type fixnum i))
      (when (= (predict-digit network (aref test-images i))
               (aref test-labels i))
        (incf correct))
      (when (zerop (mod i 1000))
        (format t "Evaluated ~D/~D examples~%" i total)))
    (let ((accuracy (* 100.0 (/ correct total))))
      (declare (type single-float accuracy))
      (format t "Test Accuracy: ~,2F% (~D/~D correct)~%" accuracy correct total)
      accuracy)))

(defun confusion-matrix (network test-images test-labels)
  "Generate confusion matrix for predictions"
  (declare (type neural-network network)
           (type (simple-array t (*)) test-images)
           (type (simple-array (unsigned-byte 8) (*)) test-labels))
  (let ((matrix (make-array '(10 10) :element-type 'fixnum :initial-element 0))
        (total (length test-images)))
    (declare (type fixnum total))
    (dotimes (i total)
      (declare (type fixnum i))
      (let ((predicted (predict-digit network (aref test-images i)))
            (actual (aref test-labels i)))
        (declare (type fixnum predicted actual))
        (incf (aref matrix actual predicted))))
    
    (format t "~%Confusion Matrix (rows=actual, cols=predicted):~%")
    (format t "    ")
    (dotimes (i 10)
      (format t "~3D " i))
    (format t "~%")
    
    (dotimes (i 10)
      (format t "~2D: " i)
      (dotimes (j 10)
        (format t "~3D " (aref matrix i j)))
      (format t "~%"))
    
    matrix))

(defun save-network (network filename)
  "Save trained network to file"
  (declare (type neural-network network)
           (type string filename))
  (with-open-file (stream filename
                   :direction :output
                   :if-exists :supersede
                   :if-does-not-exist :create)
    (format stream ";; MNIST OCR Neural Network~%")
    (format stream ";; Saved on ~A~%~%" (get-universal-time))
    
    (format stream "(defparameter *saved-network*~%")
    (format stream "  (make-neural-network~%")
    
    (format stream "   :w1 (make-array '(~D ~D) :element-type 'single-float :initial-contents~%"
            (array-dimension (neural-network-w1 network) 0)
            (array-dimension (neural-network-w1 network) 1))
    (format stream "        '~S)~%" (neural-network-w1 network))
    
    (format stream "   :b1 (make-array ~D :element-type 'single-float :initial-contents~%"
            (length (neural-network-b1 network)))
    (format stream "        '~S)~%" (neural-network-b1 network))
    
    (format stream "   :w2 (make-array '(~D ~D) :element-type 'single-float :initial-contents~%"
            (array-dimension (neural-network-w2 network) 0)
            (array-dimension (neural-network-w2 network) 1))
    (format stream "        '~S)~%" (neural-network-w2 network))
    
    (format stream "   :b2 (make-array ~D :element-type 'single-float :initial-contents~%"
            (length (neural-network-b2 network)))
    (format stream "        '~S)))~%" (neural-network-b2 network)))
  
  (format t "Network saved to ~A~%" filename))

(defun load-network (filename)
  "Load network from file"
  (declare (type string filename))
  (with-open-file (stream filename :direction :input)
    (let ((*read-eval* t))
      (load stream)
      (symbol-value '*saved-network*))))

(defun visualize-predictions (network test-images test-labels &optional (num-samples 10))
  "Show predictions for a few test samples"
  (declare (type neural-network network)
           (type (simple-array t (*)) test-images)
           (type (simple-array (unsigned-byte 8) (*)) test-labels)
           (type fixnum num-samples))
  (format t "~%Sample Predictions:~%")
  (format t "==================~%")
  
  (dotimes (i (min num-samples (length test-images)))
    (declare (type fixnum i))
    (let* ((image (aref test-images i))
           (actual (aref test-labels i))
           (predicted (predict-digit network image))
           (normalized (normalize-image image))
           (output (forward-pass network normalized)))
      
      (format t "~%Sample ~D:~%" i)
      (format t "  Actual label: ~D~%" actual)
      (format t "  Predicted: ~D~%" predicted)
      (format t "  Confidence: ~,2F%~%" (* 100.0 (aref output predicted)))
      
      (when (/= actual predicted)
        (format t "  *** MISCLASSIFIED ***~%"))
      
      (format t "  Top 3 predictions:~%")
      (let ((sorted-indices (sort (loop for i below 10 collect i)
                                  #'> :key (lambda (idx) (aref output idx)))))
        (dotimes (j 3)
          (let ((idx (nth j sorted-indices)))
            (format t "    ~D: ~,2F%~%" idx (* 100.0 (aref output idx)))))))))