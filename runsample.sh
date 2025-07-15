#!/bin/bash

CMD=$(cat<<EOF
(require :asdf)
(push #P"./" asdf:*central-registry*)
(asdf:load-system :mnist-ocr)
(in-package :mnist-ocr)
EOF
)

echo "${CMD}" | sbcl --load mnist-ocr.asd
