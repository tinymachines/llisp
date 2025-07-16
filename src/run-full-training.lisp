#!/usr/bin/sbcl --script

(require :asdf)
(push #P"./" asdf:*central-registry*)
(asdf:load-system :mnist-ocr)
(in-package :mnist-ocr)

;; Run the full training example
(run-mnist-training)