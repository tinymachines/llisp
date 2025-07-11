(asdf:defsystem "mnist-ocr"
  :description "MNIST OCR from scratch in Common Lisp"
  :author "Your Name <your.email@example.com>"
  :license "MIT"
  :version "1.0.0"
  :serial t
  :components ((:file "packages")
               (:file "src/data-loader")
               (:file "src/neural-net") 
               (:file "src/training")
               (:file "src/inference")
               (:file "examples/train-mnist")))