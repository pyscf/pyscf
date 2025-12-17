; -*-Lisp-*-
;;;;
;;;; Copyright (C) 2013-  Qiming Sun <osirpt.sun@gmail.com>
;;;; Copyright (C) 2025-  Christopher Hillenbrand <chillenbrand15@gmail.com>
;;;; Description:
;;;  dump to output file

(load "utility.cl")
(load "parser.cl")
(load "derivator.cl")

(defun gen-subscript (cells-streamer raw-script)
  (labels ((gen-tex-iter (raw-script)
                         (cond ((null raw-script) raw-script)
                               ((vector? raw-script)
                                 (list (gen-tex-iter (comp-x raw-script))
                                       (gen-tex-iter (comp-y raw-script))
                                       (gen-tex-iter (comp-z raw-script))))
                               ((cells? raw-script)
                                 (funcall cells-streamer raw-script))
                               (t (mapcar cells-streamer raw-script)))))
    (gen-tex-iter raw-script)))
(defun flatten-raw-script (raw-script)
  (let ((terms ()))
    (gen-subscript (lambda (x) (push x terms)) raw-script)
    (reverse terms)))

(defun convert-from-n-sys (ls n)
  (reduce (lambda (x y) (+ (* x n) y)) ls
    :initial-value 0))

(defun xyz-to-ternary (xyzs)
  (cond ((eql xyzs 'x) 0)
        ((eql xyzs 'y) 1)
        ((eql xyzs 'z) 2)
        (t (error " unknown subscript ~a" xyzs))))

(defun ternary-subscript (ops)
  "convert the polynomial xyz to the ternary"
  (cond ((null ops) ops)
        (t (convert-from-n-sys (mapcar #'xyz-to-ternary
                                   (remove-if (lambda (x) (eql x 's))
                                       (scripts-of ops)))
                               3))))
(defun cell-converter-cplx (cell fout tot-bits &optional (s-symb "s"))
  (let* ((fac (realpart (phase-of cell)))
        (const@3 (ternary-subscript (consts-of cell)))
        (op@3 (ternary-subscript (ops-of cell)))
        (op_idx (if (null op@3) 0 op@3))
        (const_str (if (null const@3) "" (format nil "c[~a]*" const@3)))
        (fac_str (cond ((equal fac 1) " + ")
                        ((equal fac -1) " - ")
                        ((< fac 0) (format nil " ~a * " fac))
                        (t (format nil " + ~a * " fac))))
        )
        (format fout "~a~a~a\[~a\]" fac_str const_str s-symb op_idx)
        ))

; we absorbed the factors into the sR and sI already,
; see cell-converter-ft.
(defun cell-nofac-sum (cell fout tot-bits &optional (s-symb "s"))
  (let* ((fac (realpart (phase-of cell)))
        (const@3 (ternary-subscript (consts-of cell)))
        (op@3 (ternary-subscript (ops-of cell)))
        (op_idx (if (null op@3) 0 op@3))
        (const_str (if (null const@3) "" (format nil "c[~a]*" const@3)))
        )
        (format fout " + ~a~a\[~a\]" const_str s-symb op_idx)
        ))

(defun cell-converter-ft (cell fout tot-bits &optional (s-symb "s"))
  (let* ((fac (realpart (phase-of cell)))
         (const@3 (ternary-subscript (consts-of cell)))
         (op@3 (ternary-subscript (ops-of cell)))
         (idx (if (null op@3) 0 op@3))
         (xyzbin (s-for-oneroot-i tot-bits idx)))
    (destructuring-bind (xbin ybin zbin) xyzbin
      (format fout "ZMAD_MUL(sR[~a], sI[~a], g~a, g~a, g~a, ~a);~%"
         idx idx xbin ybin zbin (float fac))
        )))

(defun to-c-code-string (fout c-converter flat-script tot-bits &optional (s-symb "s"))
  (flet ((c-streamer (cs)
                     (with-output-to-string (tmpout)
                       (cond ((null cs) (format tmpout " 0"))
                             ((cell? cs) (funcall c-converter cs tmpout tot-bits s-symb))
                             (t (mapcar (lambda (c) (funcall c-converter c tmpout tot-bits s-symb)) cs))))))
    (mapcar #'c-streamer flat-script)))


(defun gen-c-block-core (fout flat-script tot-bits pm_or_add)
  (format fout "for (n = 0; n < nf; n++) {~%")
  (format fout "ix = idx[0+n*3];~%")
  (format fout "iy = idx[1+n*3];~%")
  (format fout "iz = idx[2+n*3];~%")
  (format fout "#pragma GCC ivdep~%")
  (format fout "for (k = 0; k < bs; k++) {~%")
  (let ((assemb (to-c-code-string fout #'cell-converter-ft flat-script tot-bits))
        (comp (length flat-script)))
    (format fout "double sR[~a] = {0};~%" (expt 3 tot-bits))
    (format fout "double sI[~a] = {0};~%" (expt 3 tot-bits))
    (loop for s in assemb
          for gid from 0 do
            (format fout "~a" s)))
  (let ((assemb (to-c-code-string fout #'cell-nofac-sum flat-script tot-bits "sR"))
      (comp (length flat-script)))
    (loop for s in assemb
          for gid from 0 do
            (format fout "goutR[(n*~a+~a)*bs+k] ~a ~a;~%" comp gid pm_or_add s)))
  (let ((assemb (to-c-code-string fout #'cell-nofac-sum flat-script tot-bits "sI"))
      (comp (length flat-script)))
    (loop for s in assemb
          for gid from 0 do
            (format fout "goutI[(n*~a+~a)*bs+k] ~a ~a;~%" comp gid pm_or_add s)))
  (format fout "}~%}~%"))



(defun gen-c-block (fout flat-script tot-bits)
  (gen-c-block-core fout flat-script tot-bits "=")
  )

(defun gen-c-block+ (fout flat-script tot-bits)
  (gen-c-block-core fout flat-script tot-bits "+=")
  )

(defun gen-c-block-with-empty (fout flat-script tot-bits)
  (format fout "if (empty) {~%")
  (gen-c-block fout flat-script tot-bits)
  (format fout "} else {~%")
  (gen-c-block+ fout flat-script tot-bits)
  (format fout "}"))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; effective keys are p,r,ri,...
(defun effect-keys (ops)
  (remove-if-not (lambda (x)
                   (or (member x *act-left-right*)
                       (member x *intvar*)))
      ops))
(defun g?e-of (key)
  (case key
    ((p ip nabla px py pz p* ip* nabla* px* py* pz*) "D_")
    ((r x y z) "R_") ; the vector origin is on the center of the basis it acts on
    ((ri rj rk rl) "RC") ; the vector origin is R[ijkl]
    ((xi xj xk xl) "RC")
    ((yi yj yk yl) "RC")
    ((zi zj zk zl) "RC")
    ((r0 x0 y0 z0 g) "R0") ; R0 ~ the vector origin is (0,0,0)
    ((rc xc yc zc) "RC") ; the vector origin is set in env[PTR_COMMON_ORIG]
    ((nabla-rinv nabla-r12 breit-r1 breit-r2) "D_")
    (otherwise (error "unknown key ~a~%" key))))

(defun dump-header (fout)
  (format fout "/*
 * Copyright (C) 2013-  Qiming Sun <osirpt.sun@gmail.com>
 * Copyright (C) 2025-  Christopher Hillenbrand <chillenbrand15@gmail.com>
 * Description: code generated by  gen-code-ft.cl
 */
#include <stdlib.h>
#include <complex.h>
#include \"cint.h\"
#include \"gto/ft_ao.h\"


#define G1E_R_I(f, g, li, lj, lk) f = g + bs * envs->g_stride_i;
#define G1E_R_J(f, g, li, lj, lk) f = g + bs * envs->g_stride_j;
#define G1E_R_K(f, g, li, lj, lk) f = g + bs * envs->g_stride_k;

#define G1E_RCI(f, g, li, lj, lk)   GTO_ft_x1i(f, g, dri, li, lj, envs);
#define G1E_RCJ(f, g, li, lj, lk)   GTO_ft_x1j(f, g, drj, li, lj, envs);

#define G1E_R0I(f, g, li, lj, lk)   GTO_ft_x1i(f, g, envs->ri, li, lj, envs);
#define G1E_R0J(f, g, li, lj, lk)   GTO_ft_x1j(f, g, envs->rj, li, lj, envs);

#define ZMAD_MUL(outR, outI, gx, gy, gz, factor) \\
        xyR = gx##R[ix*bs+k] * gy##R[iy*bs+k] - gx##I[ix*bs+k] * gy##I[iy*bs+k]; \\
        xyI = gx##R[ix*bs+k] * gy##I[iy*bs+k] + gx##I[ix*bs+k] * gy##R[iy*bs+k]; \\
        outR += factor * (xyR * gz##R[iz*bs+k] - xyI * gz##I[iz*bs+k]); \\
        outI += factor * (xyR * gz##I[iz*bs+k] + xyI * gz##R[iz*bs+k]);

"))

(defun dump-declare-dri-for-rc (fout i-ops symb)
  (when (intersection '(rc xc yc zc) i-ops)
        (format fout "double dr~a[3];~%" symb)
        (format fout "dr~a[0] = envs->r~a[0] - envs->env[PTR_COMMON_ORIG+0];~%" symb symb)
        (format fout "dr~a[1] = envs->r~a[1] - envs->env[PTR_COMMON_ORIG+1];~%" symb symb)
        (format fout "dr~a[2] = envs->r~a[2] - envs->env[PTR_COMMON_ORIG+2];~%" symb symb))
  (when (intersection '(ri xi yi zi) i-ops)
        (if (intersection '(rc xc yc zc) i-ops)
            (error "Cannot declare dri because rc and ri coexist"))
        (format fout "double dr~a[3];~%" symb)
        (format fout "dr~a[0] = envs->r~a[0] - envs->ri[0];~%" symb symb)
        (format fout "dr~a[1] = envs->r~a[1] - envs->ri[1];~%" symb symb)
        (format fout "dr~a[2] = envs->r~a[2] - envs->ri[2];~%" symb symb))
  (when (intersection '(rj xj yj zj) i-ops)
        (if (intersection '(rc xc yc zc) i-ops)
            (error "Cannot declare drj because rc and rj coexist"))
        (format fout "double dr~a[3];~%" symb)
        (format fout "dr~a[0] = envs->r~a[0] - envs->rj[0];~%" symb symb)
        (format fout "dr~a[1] = envs->r~a[1] - envs->rj[1];~%" symb symb)
        (format fout "dr~a[2] = envs->r~a[2] - envs->rj[2];~%" symb symb))
  (when (intersection '(rk xk yk zk) i-ops)
        (if (intersection '(rc xc yc zc) i-ops)
            (error "Cannot declare drk because rc and rk coexist"))
        (format fout "double dr~a[3];~%" symb)
        (format fout "dr~a[0] = envs->r~a[0] - envs->rk[0];~%" symb symb)
        (format fout "dr~a[1] = envs->r~a[1] - envs->rk[1];~%" symb symb)
        (format fout "dr~a[2] = envs->r~a[2] - envs->rk[2];~%" symb symb))
  (when (intersection '(rl xl yl zl) i-ops)
        (if (intersection '(rc xc yc zc) i-ops)
            (error "Cannot declare drl because rc and rl coexist"))
        (format fout "double dr~a[3];~%" symb)
        (format fout "dr~a[0] = envs->r~a[0] - envs->rl[0];~%" symb symb)
        (format fout "dr~a[1] = envs->r~a[1] - envs->rl[1];~%" symb symb)
        (format fout "dr~a[2] = envs->r~a[2] - envs->rl[2];~%" symb symb)))

(defun dump-declare-giao-ij (fout bra ket)
  (let ((n-giao (count 'g (append bra ket))))
    (when (> n-giao 0)
          (format fout "double rirj[3], c[~a];~%" (expt 3 n-giao))
          (format fout "rirj[0] = envs->ri[0] - envs->rj[0];~%")
          (format fout "rirj[1] = envs->ri[1] - envs->rj[1];~%")
          (format fout "rirj[2] = envs->ri[2] - envs->rj[2];~%")
          (loop
         for i upto (1- (expt 3 n-giao)) do
           (format fout "c[~a] = 1" i)
           (loop
          for j from (1- n-giao) downto 0
            and res = i then (multiple-value-bind (int res) (floor res (expt 3 j))
                               (format fout " * rirj[~a]" int)
                               res))
           (format fout ";~%")))))

; l-combo searches op_bit from left to right
;  o100 o010 o001|g...>  =>  |g...> = o100 |g0..>
; a operator can only be applied to the left of the existed ones
;  |g100,l> = o100 |g000,l+1>
;  |g101,l> = o100 |g001,l+1>
;  |g110,l> = o100 |g010,l+1>
;  |g111,l> = o100 |g011,l+1>
; as a result, g* intermediates are generated from the previous one whose
; id (in binary) can be obtained by removing the first bit 1 from current
; id (in binary), see combo-bra function, eg
;  000  g0
;  001  g1 from g0 (000)
;  010  g2 from g0 (000)
;  011  g3 from g1 (001)
;  100  g4 from g0 (000)
;  101  g5 from g1 (001)
;  110  g6 from g2 (010)
;  111  g7 from g3 (011)
; r-combo searches op_bit from right to left
;  o100 o010 o001|g...>  =>  |g...> = o001 |g..0>
; a operator can only be applied to the right of the exsited ones
;  |g100,l+2> = o100 |g000,l+3>
;  |g101,l  > = o001 |g100,l+1>
;  |g110,l+1> = o010 |g100,l+2>
;  |g111,l  > = o001 |g110,l+1>
; as a result, g* intermediates are generated from the previous one whose
; id (in binary) can be obtained by removing the last bit 1 from current
; id (in binary), see combo-ket function, eg
;  000  g0
;  001  g1 from g0 (000)
;  010  g2 from g0 (000)
;  011  g3 from g2 (010)
;  100  g4 from g0 (000)
;  101  g5 from g4 (100)
;  110  g6 from g4 (100)
;  111  g7 from g6 (110)
; [lr]-combo have no connection with <bra| or |ket>
;    def l_combinator(self, ops, ig, mask, template):
(defun first-bit1 (n)
  (loop
 for i upto 31
   thereis (if (zerop (ash n (- i))) (1- i))))
(defun last-bit1 (n)
  ; how many 0s follow the last bit 1
  (loop
 for i upto 31
   thereis (if (oddp (ash n (- i))) i)))
(defun combo-bra (fout fmt ops-rev n-ops ig mask)
  (let* ((right (first-bit1 (ash ig (- mask))))
         (left (- n-ops right 1))
         (ig0 (- ig (ash 1 (+ mask right))))
         (op (nth right ops-rev)))
    (format fout fmt (g?e-of op) ig ig0 left)))
(defun combo-ket (fout fmt ops-rev i-len ig mask)
  (let* ((right (last-bit1 (ash ig (- mask))))
         (ig0 (- ig (ash 1 (+ mask right))))
         (op (nth right ops-rev)))
    (format fout fmt (g?e-of op) ig ig0 i-len right)))
(defun combo-opj (fout fmt-op fmt-j opj-rev i-len j-len ig mask)
  (let ((right (last-bit1 (ash ig (- mask)))))
    (if (< right j-len) ; does not reach op yet
        (combo-ket fout fmt-j opj-rev i-len ig mask)
        (let ((ig0 (- ig (ash 1 (+ mask right))))
              (op (nth right opj-rev)))
          (if (member op *act-left-right*)
              (format fout fmt-op
                (g?e-of op) ig ig0 i-len right
                (g?e-of op) (1+ ig) ig0 i-len right
                ig (1+ ig))
              (if (and (intersection *act-left-right* opj-rev)
                       (< (1+ right) (length opj-rev))) ; ops have *act-left-right* but the rightmost op is not
                  (format fout fmt-j (g?e-of op) ig ig0 (1+ i-len) right)
                  (format fout fmt-j (g?e-of op) ig ig0 i-len right)))))))

(defun power2-range (n &optional (shift 0))
  (range (+ shift (ash 1 n)) (+ shift (ash 1 (1+ n)))))
(defun dump-combo-braket (fout fmt-i fmt-op fmt-j i-rev op-rev j-rev mask)
  (let* ((i-len (length i-rev))
         (j-len (length j-rev))
         (op-len (length op-rev))
         (opj-rev (append j-rev op-rev)))
    (loop
   for right from mask to (+ mask j-len op-len -1) do
     (loop
    for ig in (power2-range right) do
      (combo-opj fout fmt-op fmt-j opj-rev i-len j-len ig mask)))
    (let ((shft (+ op-len j-len mask)))
      (loop
     for right from shft to (+ shft i-len -1) do
       (loop
      for ig in (power2-range right) do
        (combo-bra fout fmt-i i-rev i-len ig shft))))))

(defun dec-to-ybin (n)
  (parse-integer (substitute #\0 #\2 (write-to-string n :base 3))
                 :radix 2))
(defun dec-to-zbin (n)
  (parse-integer (substitute #\1 #\2
                     (substitute #\0 #\1
                         (write-to-string n :base 3)))
                 :radix 2))

;;; g-intermediates are g0, g1, g2, ...
(defun num-g-intermediates (tot-bits op i-len j-len)
  (if (and (intersection *act-left-right* op)
           ; nabla-rinv is the last one in the operation list
           (<= (+ i-len (length op) j-len) 1))
      (ash 1 tot-bits)
      (1- (ash 1 tot-bits))))


;!!! Be very cautious of the reverse order on i-operators and k operators!
;!!! When multiple tensor components (>= rank 2) provided by the operators
;!!! on bra functions, the ordering of the multiple tensor components are
;!!! also reversed in the generated integral code
(defun gen-code-gout1e (fout intname raw-infix flat-script)
  (destructuring-bind (op bra-i ket-j bra-k ket-l)
      (split-int-expression raw-infix)
    (let* ((i-rev (effect-keys bra-i)) ;<i| already in reverse order
                                      (j-rev (reverse (effect-keys ket-j)))
                                      (op-rev (reverse (effect-keys op)))
                                      (i-len (length i-rev))
                                      (j-len (length j-rev))
                                      (op-len (length op-rev))
                                      (tot-bits (+ i-len j-len op-len))
                                      (goutinc (length flat-script)))
      (format fout "static void inner_prod_~a" intname)
      (format fout "(double *gout, double *g, int *idx, FTEnvVars *envs, int empty) {
int nf = envs->nf;
int bs = envs->block_size;
size_t g_size = envs->g_size * bs;
double *g0R = g;
double *g0I = g0R + g_size * 3;~%")
      (loop
     for i in (range (num-g-intermediates tot-bits op i-len j-len)) do
       (format fout "double *g~aR = g~aI + g_size * 3;~%" (1+ i) i)
       (format fout "double *g~aI = g~aR + g_size * 3;~%" (1+ i) (1+ i)))
      (format fout "double *goutR = gout;~%")
      (format fout "double *goutI = goutR + nf * bs * ~a;~%" goutinc)
      (format fout "int ix, iy, iz, n, k;~%")
      (dump-declare-dri-for-rc fout bra-i "i")
      (dump-declare-dri-for-rc fout (append op ket-j) "j")
      (dump-declare-giao-ij fout bra-i (append op ket-j))
      (format fout "double xyR, xyI;~%")
      ;;; generate g_(bin)
      ;;; for the operators act on the |ket>, the reversed scan order and r_combinator
      ;;; is required; for the operators acto on the <bra|, the normal scan order and
      (let ((fmt-i "G1E_~aI(g~aR, g~aR, envs->i_l+~a, envs->j_l, 0);~%")
            (fmt-op (mkstr "G1E_~aJ(g~aR, g~aR, envs->i_l+~d, envs->j_l+~a, 0);
G1E_~aI(g~aR, g~aR, envs->i_l+~d, envs->j_l+~a, 0);
for (ix = 0; ix < envs->g_size * 3; ix++) {g~aR[ix] += g~aR[ix];}~%"))
            (fmt-j (mkstr "G1E_~aJ(g~aR, g~aR, envs->i_l+~d, envs->j_l+~a, 0);~%")))
        (dump-combo-braket fout fmt-i fmt-op fmt-j i-rev op-rev j-rev 0))

      (let ((fmt-i "G1E_~aI(g~aI, g~aI, envs->i_l+~a, envs->j_l, 0);~%")
            (fmt-op (mkstr "G1E_~aJ(g~aI, g~aI, envs->i_l+~d, envs->j_l+~a, 0);
G1E_~aI(g~aI, g~aI, envs->i_l+~d, envs->j_l+~a, 0);
for (ix = 0; ix < envs->g_size * 3; ix++) {g~a[ix] += g~a[ix];}~%"))
            (fmt-j (mkstr "G1E_~aJ(g~aI, g~aI, envs->i_l+~d, envs->j_l+~a, 0);~%")))
        (dump-combo-braket fout fmt-i fmt-op fmt-j i-rev op-rev j-rev 0))


      (gen-c-block-with-empty fout flat-script tot-bits)
      (format fout "}~%")

      goutinc)))

(defun gen-code-int1e (fout intname raw-infix)
  (destructuring-bind (op bra-i ket-j bra-k ket-l)
      (split-int-expression raw-infix)
    (let* ((i-rev (effect-keys bra-i)) ;<i| already in reverse order
                                      (j-rev (reverse (effect-keys ket-j)))
                                      (op-rev (reverse (effect-keys op)))
                                      (i-len (length i-rev))
                                      (j-len (length j-rev))
                                      (op-len (length op-rev))
                                      (tot-bits (+ i-len j-len op-len))
                                      (raw-script (eval-int raw-infix))
                                      (flat-script (flatten-raw-script (last1 raw-script)))
                                      (ts (car raw-script))
                                      (sf (cadr raw-script))
                                      (goutinc (length flat-script))
                                      (e1comps (if (eql sf 'sf) 1 4))
                                      (tensors (if (eql sf 'sf) goutinc (/ goutinc 4)))
                                      (int1e-type (cond ((member 'nuc raw-infix) 2)
                                                        ((or (member 'rinv raw-infix)
                                                             (member 'nabla-rinv raw-infix)) 1)
                                                        (t 0)))
                                      (ngdef (with-output-to-string (tmpout)
                                               (if (or (member 'nuc raw-infix)
                                                       (member 'rinv raw-infix)
                                                       (member 'nabla-rinv raw-infix))
                                                   (if (intersection *act-left-right* op)
                                                       (format tmpout "int ng[] = {~d, ~d, 0, 0, ~d, ~d, 0, ~d};~%"
                                                         (1+ i-len) (+ op-len j-len) tot-bits e1comps tensors)
                                                       (format tmpout "int ng[] = {~d, ~d, 0, 0, ~d, ~d, 0, ~d};~%"
                                                         i-len (+ op-len j-len) tot-bits e1comps tensors))
                                                   (format tmpout "int ng[] = {~d, ~d, 0, 0, ~d, ~d, 1, ~d};~%"
                                                     i-len (+ op-len j-len) tot-bits e1comps tensors))))
                                      (envs-common (with-output-to-string (tmpout)
                                                     (format tmpout ngdef)
                                                     (format tmpout "FTEnvVars envs;~%")
                                                     (format tmpout "GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size, atm, natm, bas, nbas, env);~%")
                                                     (format tmpout "envs.f_gout = &inner_prod_~a;~%" intname)
                                                     ; (unless (eql (factor-of raw-infix) 1)
                                                     ;   (format tmpout "envs.common_factor *= ~a;~%" (factor-of raw-infix)))
                            )))
      ; (write-functype-header
      ;   t intname (format nil "/* <~{~a ~}i|~{~a ~}|~{~a ~}j> */" bra-i op ket-j))
      (format fout "/* <~{~a ~}i|~{~a ~}|~{~a ~}j> */~%" bra-i op ket-j)
      (cond ((or (member 'nuc raw-infix)
                 (member 'rinv raw-infix)
                 (member 'nabla-rinv raw-infix))
              (format fout "Error, gout1e-rinv not supported yet~%"))
            (t (gen-code-gout1e fout intname raw-infix flat-script)))

      (format fout "~%~%")
      (format fout "int GTO_ft_~a_cart(double *outR, double *outI, int *shls, int *dims,
FPtr_eval_gz eval_gz, double complex fac,
double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
int *atm, int natm, int *bas, int nbas, double *env, double *cache) {~%" intname)
      (format fout envs-common)

      (format fout "return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_cart, &envs);~%}~%~%")

      ;;; _sph
      (format fout "int GTO_ft_~a_sph(double *outR, double *outI, int *shls, int *dims,
FPtr_eval_gz eval_gz, double complex fac,
double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
int *atm, int natm, int *bas, int nbas, double *env, double *cache) {~%" intname)
      (format fout envs-common)
      (format fout "return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_sph, &envs);~%}~%~%"))))


(defun s-for-oneroot-i (tot-bits i)
  (let* ((i_num (if (null i) 0 i))
         (ybin (dec-to-ybin i_num))
         (zbin (dec-to-zbin i_num))
         (xbin (- (ash 1 tot-bits) 1 ybin zbin)))
    (list xbin ybin zbin)))



(defun gen-ftao (filename &rest items)
  "sp can be one of 'spinor 'spheric 'cart"
  (with-open-file (fout (mkstr filename)
                        :direction :output :if-exists :supersede)
    (dump-header fout)
  (flet ((gen-code (item)
                   (let ((intname (mkstr (car item)))
                         (raw-infix (cadr item)))
                     (cond ((one-electron-int? raw-infix)
                             (gen-code-int1e fout intname raw-infix))))))
    (mapcar #'gen-code items))))

; gcl -load sigma.o -batch -eval "( .. )"

;; vim: ft=lisp
