#!/usr/bin/sbcl --script

(require :asdf)

(push #P"./" asdf:*central-registry*)

(asdf:load-system :mnist-ocr)

(in-package :mnist-ocr)

(format t "~%Running quick MNIST training test...~%")
(format t "=====================================~%~%")

(handler-case
    (quick-train :epochs 2 :learning-rate 0.1 :hidden-size 64)
  (error (e)
    (format t "Error: ~A~%" e)))