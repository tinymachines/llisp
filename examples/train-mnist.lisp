(in-package :mnist-ocr)

(defun run-mnist-training ()
  "Complete example of training MNIST from scratch"
  (format t "~%========================================~%")
  (format t "MNIST OCR Training Example~%")
  (format t "========================================~%~%")
  
  (handler-case
      (multiple-value-bind (train-images train-labels test-images test-labels)
          (load-mnist-data "data/")
        
        (format t "~%Data loaded successfully!~%")
        (format t "Training set: ~D images~%" (length train-images))
        (format t "Test set: ~D images~%~%" (length test-images))
        
        (format t "Creating neural network...~%")
        (format t "Architecture: 784 → 128 → 10~%~%")
        (let ((network (create-network *input-size* *hidden-size* *output-size*)))
          
          (format t "Initial test accuracy (before training):~%")
          (evaluate-accuracy network test-images test-labels)
          
          (format t "~%Starting training...~%")
          (format t "Epochs: 10~%")
          (format t "Learning rate: 0.1~%~%")
          
          (time (train-network network train-images train-labels 10 0.1))
          
          (format t "~%Training complete!~%~%")
          (format t "Final test accuracy:~%")
          (evaluate-accuracy network test-images test-labels)
          
          (format t "~%Generating confusion matrix...~%")
          (confusion-matrix network test-images test-labels)
          
          (format t "~%Showing sample predictions...~%")
          (visualize-predictions network test-images test-labels 5)
          
          (format t "~%Saving trained network...~%")
          (save-network network "trained-mnist-net.lisp")
          
          (format t "~%Training session complete!~%")
          (format t "========================================~%")
          
          network))
    
    (file-error (e)
      (format t "~%ERROR: Could not load MNIST data files!~%")
      (format t "Error details: ~A~%~%" e)
      (format t "Please ensure the MNIST dataset files are in the 'data/' directory:~%")
      (format t "  - train-images-idx3-ubyte~%")
      (format t "  - train-labels-idx1-ubyte~%")
      (format t "  - t10k-images-idx3-ubyte~%")
      (format t "  - t10k-labels-idx1-ubyte~%~%")
      (format t "Download instructions are in the README.md file.~%"))
    
    (error (e)
      (format t "~%ERROR: ~A~%" e))))

(defun quick-train (&key (epochs 5) (learning-rate 0.1) (hidden-size 128))
  "Quick training function with customizable parameters"
  (format t "Quick training with:~%")
  (format t "  Epochs: ~D~%" epochs)
  (format t "  Learning rate: ~F~%" learning-rate)
  (format t "  Hidden layer size: ~D~%~%" hidden-size)
  
  (handler-case
      (multiple-value-bind (train-images train-labels test-images test-labels)
          (load-mnist-data "data/")
        
        (let ((network (create-network *input-size* hidden-size *output-size*)))
          
          (setf *print-training-progress* nil)
          
          (format t "Training...~%")
          (time (train-network network 
                               (subseq train-images 0 10000)
                               (subseq train-labels 0 10000)
                               epochs 
                               learning-rate))
          
          (setf *print-training-progress* t)
          
          (format t "~%Evaluating on full test set...~%")
          (evaluate-accuracy network test-images test-labels)
          
          network))
    
    (error (e)
      (format t "Error during training: ~A~%" e))))

(defun test-single-image (network-or-filename image-index)
  "Test network on a single image from the test set"
  (let ((network (if (stringp network-or-filename)
                     (load-network network-or-filename)
                     network-or-filename)))
    
    (handler-case
        (multiple-value-bind (train-images train-labels test-images test-labels)
            (load-mnist-data "data/")
          (declare (ignore train-images train-labels))
          
          (if (< image-index (length test-images))
              (let* ((image (aref test-images image-index))
                     (actual (aref test-labels image-index))
                     (predicted (predict-digit network image))
                     (normalized (normalize-image image))
                     (output (forward-pass network normalized)))
                
                (format t "~%Testing image ~D:~%" image-index)
                (format t "Actual label: ~D~%" actual)
                (format t "Predicted: ~D~%" predicted)
                (format t "Result: ~A~%~%" (if (= actual predicted) "CORRECT" "INCORRECT"))
                
                (format t "Network output probabilities:~%")
                (dotimes (i 10)
                  (format t "  ~D: ~,2F%~%" i (* 100.0 (aref output i))))
                
                (display-image image)
                
                predicted)
              (format t "Image index ~D out of range (max: ~D)~%" 
                      image-index (1- (length test-images)))))
      
      (error (e)
        (format t "Error: ~A~%" e)))))

(defun display-image (image)
  "Display MNIST image as ASCII art"
  (format t "~%Image visualization:~%")
  (dotimes (row 28)
    (dotimes (col 28)
      (let ((pixel (aref image (+ (* row 28) col))))
        (cond
          ((< pixel 50) (format t "  "))
          ((< pixel 100) (format t ".."))
          ((< pixel 150) (format t "++"))
          ((< pixel 200) (format t "##"))
          (t (format t "@@")))))
    (format t "~%")))