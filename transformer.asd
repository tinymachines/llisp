(defsystem "transformer"
  :description "A simple transformer implementation in Common Lisp"
  :author "Your Name"
  :license "MIT"
  :version "0.1.0"
  :depends-on ()
  :serial t
  :components ((:file "transformer-package")
               (:module "src"
                :components ((:file "math-utils")
                             (:file "attention")
                             (:file "transformer-block")
                             (:file "transformer-model")
                             (:file "dataset")
                             (:file "transformer-training")))
               (:module "examples"
                :components ((:file "train-addition")))))