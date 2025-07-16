(in-package :transformer)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defun make-addition-dataset ()
  "Generate dataset for learning addition of two 2-digit numbers"
  (let ((data '()))
    ;; Generate all possible additions of 2-digit numbers
    (loop for i from 0 below 100
          do (loop for j from 0 below 100
                   do (let* ((sum (+ i j))
                             (example (list (floor i 10)     ; tens digit of first number
                                            (mod i 10)        ; ones digit of first number
                                            (floor j 10)      ; tens digit of second number
                                            (mod j 10)        ; ones digit of second number
                                            (floor sum 100)   ; hundreds digit of sum
                                            (mod (floor sum 10) 10) ; tens digit of sum
                                            (mod sum 10))))   ; ones digit of sum
                        (push example data))))
    
    ;; Shuffle the dataset
    (let ((array (make-array (length data) :initial-contents (nreverse data))))
      ;; Fisher-Yates shuffle
      (loop for i from (1- (length array)) downto 1
            do (let ((j (random (1+ i))))
                 (rotatef (aref array i) (aref array j))))
      
      ;; Split into train and test sets
      (let* ((total-size (length array))
             (train-size 8000)
             (test-size (- total-size train-size))
             (x-train (make-array (list train-size 6) :element-type 'fixnum))
             (y-train (make-array (list train-size 6) :element-type 'fixnum))
             (x-test (make-array (list test-size 6) :element-type 'fixnum))
             (y-test (make-array (list test-size 6) :element-type 'fixnum)))
        
        ;; Fill training data
        (dotimes (i train-size)
          (let ((example (aref array i)))
            ;; X contains first 6 elements (input sequence)
            (dotimes (j 6)
              (setf (aref x-train i j) (nth j example)))
            ;; Y contains elements 1-7 (shifted output sequence)
            (dotimes (j 6)
              (setf (aref y-train i j) (nth (1+ j) example)))))
        
        ;; Fill test data
        (dotimes (i test-size)
          (let ((example (aref array (+ train-size i))))
            (dotimes (j 6)
              (setf (aref x-test i j) (nth j example)))
            (dotimes (j 6)
              (setf (aref y-test i j) (nth (1+ j) example)))))
        
        (values x-train y-train x-test y-test)))))

(defun create-batch (data indices)
  "Create a batch from data using specified indices"
  (declare (type (simple-array fixnum (* *)) data)
           (type list indices))
  (let* ((batch-size (length indices))
         (seq-len (array-dimension data 1))
         (batch (make-array (list batch-size seq-len) :element-type 'fixnum)))
    (loop for idx in indices
          for i from 0
          do (dotimes (j seq-len)
               (setf (aref batch i j) (aref data idx j))))
    batch))

(defun get-random-batch (x-data y-data batch-size)
  "Get a random batch from the dataset"
  (declare (type (simple-array fixnum (* *)) x-data y-data)
           (type fixnum batch-size))
  (let* ((data-size (array-dimension x-data 0))
         (indices (loop for i from 0 below batch-size
                        collect (random data-size))))
    (values (create-batch x-data indices)
            (create-batch y-data indices))))