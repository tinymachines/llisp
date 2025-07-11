(defpackage #:mnist-ocr
  (:use #:cl)
  (:export #:load-mnist-data
           #:create-network
           #:train-network
           #:predict-digit
           #:evaluate-accuracy
           #:save-network
           #:load-network))