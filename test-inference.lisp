;;; Simple inference test script
(in-package :mnist-ocr)

(defun debug-inference ()
  "Debug inference step by step"
  (format t "=== MNIST OCR Inference Debug ===~%~%")
  
  ;; Load test data
  (format t "1. Loading MNIST test data...~%")
  (multiple-value-bind (train-images train-labels test-images test-labels)
      (load-mnist-data "data/")
    (format t "   Loaded ~D test images~%" (length test-images))
    
    ;; Try to load a saved network
    (format t "~%2. Loading trained network...~%")
    (let ((network (if (probe-file "trained-mnist-net.lisp")
                       (progn 
                         (format t "   Found saved network file~%")
                         (load "trained-mnist-net.lisp")
                         *saved-network*)
                       (progn
                         (format t "   No saved network found, creating random network~%")
                         (create-network 784 128 10)))))
      
      ;; Debug network structure
      (format t "~%3. Network structure:~%")
      (format t "   W1 dimensions: ~A~%" (array-dimensions (neural-network-w1 network)))
      (format t "   B1 length: ~A~%" (length (neural-network-b1 network)))
      (format t "   W2 dimensions: ~A~%" (array-dimensions (neural-network-w2 network)))
      (format t "   B2 length: ~A~%" (length (neural-network-b2 network)))
      
      ;; Test with first image
      (format t "~%4. Testing with first image...~%")
      (let* ((test-image (aref test-images 0))
             (actual-label (aref test-labels 0)))
        
        (format t "   Original image type: ~A~%" (type-of test-image))
        (format t "   Image size: ~A~%" (length test-image))
        (format t "   Actual label: ~A~%" actual-label)
        
        ;; Normalize image
        (format t "~%5. Normalizing image...~%")
        (let ((normalized (normalize-image test-image)))
          (format t "   Normalized type: ~A~%" (type-of normalized))
          (format t "   Normalized size: ~A~%" (length normalized))
          (format t "   First few values: ~A~%" 
                  (subseq normalized 0 (min 5 (length normalized))))
          
          ;; Try forward pass
          (format t "~%6. Running forward pass...~%")
          (handler-case
              (multiple-value-bind (output a1 z1)
                  (forward-pass network normalized)
                (format t "   SUCCESS!~%")
                (format t "   Output type: ~A~%" (type-of output))
                (format t "   Output size: ~A~%" (length output))
                (format t "   Output probabilities: ~A~%" output)
                
                ;; Get prediction
                (let ((predicted (argmax output)))
                  (format t "~%7. Prediction result:~%")
                  (format t "   Predicted digit: ~A~%" predicted)
                  (format t "   Confidence: ~,2F%~%" (* 100.0 (aref output predicted)))
                  (format t "   Actual digit: ~A~%" actual-label)
                  (format t "   Correct: ~A~%" (if (= predicted actual-label) "YES" "NO"))))
            (error (e)
              (format t "   ERROR: ~A~%" e)
              (format t "   This is the error you're seeing!~%")))))))

(defun quick-test ()
  "Quick test of inference"
  (format t "=== Quick Inference Test ===~%")
  
  ;; Load minimal test data (just a few images)
  (multiple-value-bind (train-images train-labels test-images test-labels)
      (load-mnist-data "data/")
    
    ;; Load or create network
    (let ((network (if (probe-file "trained-mnist-net.lisp")
                       (progn 
                         (load "trained-mnist-net.lisp")
                         *saved-network*)
                       (create-network 784 128 10))))
      
      ;; Test on first 5 images
      (format t "Testing on first 5 images:~%")
      (dotimes (i 5)
        (handler-case
            (let* ((image (aref test-images i))
                   (actual (aref test-labels i))
                   (predicted (predict-digit network image)))
              (format t "Image ~D: predicted=~D, actual=~D ~A~%" 
                      i predicted actual 
                      (if (= predicted actual) "✓" "✗")))
          (error (e)
            (format t "Image ~D: ERROR - ~A~%" i e)))))))

;; Main entry point
(defun test-inference ()
  "Main test function"
  (format t "Choose test type:~%")
  (format t "1. Debug mode (detailed)~%")
  (format t "2. Quick test~%")
  (format t "Enter choice (1 or 2): ")
  (let ((choice (read)))
    (case choice
      (1 (debug-inference))
      (2 (quick-test))
      (t (format t "Invalid choice, running debug mode~%")
         (debug-inference)))))