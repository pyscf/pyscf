#!/usr/bin/env clisp 
;;;; Copyright (C) 2015-  Qiming Sun <osirpt.sun@gmail.com>

(load "gen-code.cl")

(gen-eval "auto_eval1.c"
  '("GTOval_ig_sph"     spheric  (#C(0 1) g))
  '("GTOval_ipig_sph"   spheric  (#C(0 1) nabla g))
  '("GTOval_ig_cart"    cart     (#C(0 1) g))
)

