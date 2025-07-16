(defpackage :transformer
  (:use :cl)
  (:export 
   #:scaled-dot-product-attention
   #:multi-head-attention
   #:transformer-block
   #:transformer
   #:make-addition-dataset
   #:train-transformer
   #:forward
   #:softmax
   #:cross-entropy-loss))