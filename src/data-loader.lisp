(in-package :mnist-ocr)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defun read-u32-be (stream)
  "Read 32-bit unsigned integer in big-endian format"
  (declare (type stream stream))
  (+ (* (read-byte stream) #x1000000)
     (* (read-byte stream) #x10000)
     (* (read-byte stream) #x100)
     (read-byte stream)))

(defun load-mnist-images (filename)
  "Load MNIST image file - returns array of images"
  (with-open-file (stream filename 
                   :direction :input 
                   :element-type '(unsigned-byte 8))
    (let ((magic (read-u32-be stream))
          (num-images (read-u32-be stream))
          (rows (read-u32-be stream))
          (cols (read-u32-be stream)))
      (declare (type (unsigned-byte 32) magic num-images rows cols))
      (unless (= magic 2051)
        (error "Invalid magic number for image file: ~A" magic))
      (let ((images (make-array num-images))
            (image-size (* rows cols)))
        (declare (type fixnum image-size))
        (dotimes (i num-images)
          (declare (type fixnum i))
          (let ((image (make-array image-size 
                                   :element-type '(unsigned-byte 8))))
            (read-sequence image stream)
            (setf (aref images i) image)))
        (values images num-images rows cols)))))

(defun load-mnist-labels (filename)
  "Load MNIST label file - returns array of labels"
  (with-open-file (stream filename
                   :direction :input
                   :element-type '(unsigned-byte 8))
    (let ((magic (read-u32-be stream))
          (num-labels (read-u32-be stream)))
      (declare (type (unsigned-byte 32) magic num-labels))
      (unless (= magic 2049)
        (error "Invalid magic number for label file: ~A" magic))
      (let ((labels (make-array num-labels :element-type '(unsigned-byte 8))))
        (read-sequence labels stream)
        labels))))

(defun normalize-image (image)
  "Convert byte values to floats in [0,1]"
  (declare (type (simple-array (unsigned-byte 8) (*)) image))
  (let* ((size (length image))
         (normalized (make-array size :element-type 'single-float)))
    (declare (type fixnum size))
    (dotimes (i size)
      (declare (type fixnum i))
      (setf (aref normalized i) 
            (/ (coerce (aref image i) 'single-float) 255.0)))
    normalized))

(defun load-mnist-data (data-directory)
  "Load complete MNIST dataset from directory"
  (let ((train-images-file (merge-pathnames "train-images-idx3-ubyte" data-directory))
        (train-labels-file (merge-pathnames "train-labels-idx1-ubyte" data-directory))
        (test-images-file (merge-pathnames "t10k-images-idx3-ubyte" data-directory))
        (test-labels-file (merge-pathnames "t10k-labels-idx1-ubyte" data-directory)))
    (format t "Loading MNIST data from ~A~%" data-directory)
    (values (load-mnist-images train-images-file)
            (load-mnist-labels train-labels-file)
            (load-mnist-images test-images-file)
            (load-mnist-labels test-labels-file))))