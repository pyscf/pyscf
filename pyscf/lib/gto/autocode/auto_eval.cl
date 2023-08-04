#!/usr/bin/env clisp 
;;;; Copyright (C) 2014-2018  Qiming Sun <osirpt.sun@gmail.com>

(load "gen-code.cl")

(gen-eval "auto_eval1.c"
  '("GTOval_ig"         (#C(0 1) g))
  '("GTOval_ipig"       (#C(0 1) nabla g))
  '("GTOval_sp"         (sigma dot p))
  '("GTOval_ipsp"       (nabla sigma dot p))
  '("GTOval_ipipsp"     (nabla nabla sigma dot p))
  '("GTOval_iprc"       (nabla rc))
  '("GTOval_ipr"        (nabla r))
)

