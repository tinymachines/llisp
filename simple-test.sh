#!/bin/bash

# Simple MNIST OCR test without ASDF
echo "=== Simple MNIST OCR Test ==="
echo ""

# Check if SBCL is available
if ! command -v sbcl &> /dev/null; then
    echo "Error: SBCL not found. Please install SBCL first."
    exit 1
fi

echo "✓ SBCL found"

# Check if data exists
if [ ! -d "data" ] || [ ! -f "data/t10k-images-idx3-ubyte" ]; then
    echo "✗ MNIST data not found in data/ directory"
    echo "Please download MNIST data first"
    exit 1
fi

echo "✓ MNIST data found"
echo ""
echo "Loading and testing..."

# Create a simple test script that loads everything manually
cat > temp-test.lisp << 'EOF'
;; Load all source files manually
(load "packages.lisp")
(load "src/data-loader.lisp")
(load "src/neural-net.lisp")
(load "src/training.lisp")
(load "src/inference.lisp")

;; Simple test function
(defun simple-test ()
  (format t "=== Starting Simple Test ===~%")
  
  ;; Test 1: Load a small amount of data
  (format t "1. Loading MNIST data...~%")
  (handler-case
      (multiple-value-bind (train-images train-labels test-images test-labels)
          (mnist-ocr::load-mnist-data "data/")
        (format t "   ✓ Loaded ~D training images~%" (length train-images))
        (format t "   ✓ Loaded ~D test images~%" (length test-images))
        
        ;; Test 2: Create a network
        (format t "~%2. Creating neural network...~%")
        (let ((network (mnist-ocr::create-network 784 128 10)))
          (format t "   ✓ Network created~%")
          
          ;; Test 3: Try to normalize an image
          (format t "~%3. Testing image normalization...~%")
          (let* ((test-image (aref test-images 0))
                 (normalized (mnist-ocr::normalize-image test-image)))
            (format t "   ✓ Image normalized from ~A bytes to ~A floats~%" 
                    (length test-image) (length normalized))
            
            ;; Test 4: Try forward pass
            (format t "~%4. Testing forward pass...~%")
            (handler-case
                (multiple-value-bind (output a1 z1)
                    (mnist-ocr::forward-pass network normalized)
                  (format t "   ✓ Forward pass successful!~%")
                  (format t "   ✓ Output size: ~A~%" (length output))
                  (format t "   ✓ Output sum: ~,3F (should be ~1.0)~%" (reduce #'+ output))
                  
                  ;; Test 5: Try prediction
                  (format t "~%5. Testing prediction...~%")
                  (let ((predicted (mnist-ocr::argmax output))
                        (actual (aref test-labels 0)))
                    (format t "   ✓ Predicted digit: ~A~%" predicted)
                    (format t "   ✓ Actual digit: ~A~%" actual)
                    (format t "   ✓ Confidence: ~,1F%~%" (* 100.0 (aref output predicted))))
                  
                  (format t "~%🎉 ALL TESTS PASSED! 🎉~%")
                  (format t "Your inference system is working correctly!~%"))
              (error (e)
                (format t "   ✗ ERROR in forward pass: ~A~%" e)
                (format t "   This is likely the source of your problem.~%")))))
    (error (e)
      (format t "   ✗ ERROR loading data: ~A~%" e))))

;; Run the test
(simple-test)
EOF

# Run SBCL with the test
sbcl --script temp-test.lisp

# Clean up
rm -f temp-test.lisp

echo ""
echo "Test completed!"