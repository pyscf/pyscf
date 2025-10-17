;;;;
;;;; Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
;;;; Description:

(load "utility.cl")

(defun attach-tag (type-tag contents)
  (cons type-tag contents))
(defun type-tag (datum)
  (car datum))
(defun contents (datum)
  (cdr datum))

;;; op := (symb script)   such as (p x)
;;; op is the smallest unit in integral expression
(defun make-op (symb script)
  (list symb script))
(defun symbol-of (op)
  (car op))
(defun script-of (op)
  (cadr op))
(defun scripts-of (ops)
  "the subscripts of {op}s"
  (cond ((null ops) ops)
        (t (cons (script-of ops)
                 (scripts-of (cddr ops))))))
(defun scalar? (op)
  (and (listp op)
       (equal (script-of op) 'S)))

;;; cell is a collection of {op}s
;;; cell := (phase consts ops)
;;; * phase is a number
;;; * consts is the production of constant {op}s like (Ruv x Ruv y)
;;; * an ops is the production of {op}s
;;;   the production of {op}s is a list like (r x p x p y)
;;; * (2) is 2 x^0
;;; * vacuum cell is (0)
;;; '() is cells
(defun make-cell (phase consts ops)
  (cons phase (cons consts ops)))
(defun phase-of (cell)
  (car cell))
(defun consts-of (cell)
  (cadr cell))
(defun ops-of (cell)
  (cddr cell))
(defun string-of (cell)
  (append (consts-of cell) (ops-of cell)))
(defun cell? (c)
  (and (listp c)
       (numberp (phase-of c))))   
(defun rest-op (ops)
  (cddr ops))

;;; return a list contains only one of x/y/z
;;; _S as a scalar, does not commute with any subscripts
(defun ops-wrt-script (script str)
  (cond ((null str) str)
        ((or (eql (script-of str) script)
             (eql (script-of str) 'S))
         (cons (symbol-of str)
               (ops-wrt-script script (rest-op str))))
        (t (ops-wrt-script script (rest-op str)))))

(defmacro cell-phase-equal? (c1 c2)
  `(eql (phase-of ,c1) (phase-of ,c2)))
(defmacro cell-string-equal? (c1 c2)
  (let ((gs1 (gensym))
        (gs2 (gensym)))
    `(let ((,gs1 (string-of ,c1))
           (,gs2 (string-of ,c2)))
       (and ,@(mapcar (lambda (v)
                        `(equal (ops-wrt-script ',v ,gs1)
                                (ops-wrt-script ',v ,gs2)))
                      '(x y z))))))
(defun cell-equal? (c1 c2)
  (or (and (zerop (phase-of c1))
           (zerop (phase-of c2)))
      (and (cell-phase-equal? c1 c2)
           (cell-string-equal? c1 c2))))

;;; product of two cells, return a cell
(defun cell-multiply-num (cell n)
  (cond ((zerop n) '())
        (t (make-cell (* (phase-of cell) n)
                      (consts-of cell)
                      (ops-of cell)))))
(defun cell-multiply-cell (c1 c2)
  (cond ((null c1) '())
        ((null c2) '())
        (t (make-cell (* (phase-of c1) (phase-of c2))
                      (append (consts-of c1) (consts-of c2))
                      (append (ops-of c1) (ops-of c2))))))

;;; cells := (cell cell ...)
;;;        is the summation of several cells
;;; * vacuum cells '()  ==  a set of vacuum cell ((0 '()))  ==  0
;;;(defmacro make-cells (&rest args)
;;;  `(list ,@args))
(defun make-cells (&rest cs)
  cs)
(defun cells? (s)
  (and (listp s)
       (cell? (car s))))

;;; summation of two cells, return a cells
(defun cell-add-cell (c1 c2)
  (make-cells c1 c2))

;;; cells + cells
(defun num-add-cells (n cs)
  (let ((cell-n (make-cell n '() '())))
    (cond ((null cs) (make-cells cell-n))
          ((cell? cs) (cell-add-cell cell-n cs))
          (t (cons cell-n cs)))))
(defun cell-add-cells (c cs)
  (cond ((null cs) (make-cells c))
        ((cell? cs) (cell-add-cell c cs))
        (t (cons c cs))))
(defun cells-add-cells (cs1 cs2)
  (cond ((cell? cs1) (cell-add-cells cs1 cs2))
        ((cell? cs2) (append cs1 (make-cells cs2)))
        (t (append cs1 cs2))))

;;; cells * cells
(defun cells-multiply-num (cs n)
  (cond ((zerop n) '())
        ((cell? cs) (make-cells (cell-multiply-num cs n)))
        (t (mapcar (lambda (c) (cell-multiply-num c n)) cs))))
;;; return a cells
(defun cells-multiply-cell (cs c)
  (cond ((null c) '())
        ((cell? cs) (make-cells (cell-multiply-cell cs c)))
        (t (mapcar (lambda (c1) (cell-multiply-cell c1 c)) cs))))

(defun cells-multiply-cells (cs1 cs2)
  (cond ((null cs1) cs1)
        ((null cs2) cs2)
        ((cell? cs2) (cells-multiply-cell cs1 cs2))
        (t (cells-add-cells
             (cells-multiply-cell cs1 (car cs2))
             (cells-multiply-cells cs1 (cdr cs2))))))

#||
(defun cells-member? (c cs)
  (cond ((null cs) nil)
        (t (or (cell-string-equal? c (car cs))
               (cells-member? c (cdr cs))))))
(defun cells-member-phase-sum (c cs)
  (cond ((null cs) 0)
        ((cell-string-equal? c (car cs))
         (+ (phase-of (car cs))
            (cells-member-phase-sum c (cdr cs))))
        (t (cells-member-phase-sum c (cdr cs)))))
;; remove c from cs
(defun cells-member-wipe-off (c cs)
  (cond ((null cs) cs)
        ((cell-string-equal? c (car cs))
         (cells-member-wipe-off c (cdr cs)))
        (t (cons (car cs)
                 (cells-member-wipe-off c (cdr cs))))))
; return a cells
(defun squeeze-cells (cs)
  (cond ((null cs) cs)
        ;((cell? cs) (make-cells cs))
        ((last-one? cs) cs)
        ((last-two? cs)
         (cond ((cell-string-equal? (car cs) (cadr cs))
                (make-cells (make-cell (+ (phase-of (car cs))
                                          (phase-of (cadr cs)))
                                       (consts-of (car cs))
                                       (ops-of (car cs)))))
               (t cs)))
        (t (cons (make-cell (+ (phase-of (car cs))
                               (cells-member-phase-sum (car cs) (cdr cs)))
                            (consts-of (car cs))
                            (ops-of (car cs)))
                 (squeeze-cells (cells-member-wipe-off (car cs) (cdr cs)))))))
||#

;;; return a cells
(defun norm-ordering (cs c)
  ;;(comb (A,...) B)) = (A+B,...)    if (ops-of A) == (ops-of B)
  ;;                 or (A,...,B)    if (ops-of A) != (ops-of B)
  (cond ((cell-string-equal? (car cs) c)
         (cons (make-cell (+ (phase-of (car cs)) (phase-of c))
                          (consts-of c)
                          (ops-of c))
               (cdr cs)))
        (t (append cs (make-cells c)))))
(defun squeeze-cells (cs)
  (cond ((null cs) cs)
        ((last-one? cs) cs)
        (t (let ((squeeze-first (reduce #'norm-ordering (cdr cs)
                                        :initial-value (make-cells (car cs)))))
             (cons (car squeeze-first)
                   (squeeze-cells (cdr squeeze-first)))))))

;;; return a cells
(defun remove-vacuum-cell (cs)
  (cond ((null cs) cs)
        (t (remove-if (lambda (c) (zerop (phase-of c)))
                      cs))))

(defun cells-sum (cs-list)
  (remove-vacuum-cell
    (squeeze-cells
      (cond ((cell? cs-list) (make-cells cs-list))
            (t (reduce #'cells-add-cells cs-list))))))

(defun cells-prod (cs-list)
  (remove-vacuum-cell
    (squeeze-cells
      (cond ((cell? cs-list) (make-cells cs-list))
            (t (reduce #'cells-multiply-cells cs-list))))))

;;; tissue := (cells cells ...)
;;;

;;; quaternion = (1, sigma_x, sigma_y, sigma_z)
(defun make-quat (comp1 x y z)
  (attach-tag 'quaternion (list comp1 x y z)))
(defun sigma-1 (q)
  (car (contents q)))
(defun sigma-x (q)
  (cadr (contents q)))
(defun sigma-y (q)
  (caddr (contents q)))
(defun sigma-z (q)
  (cadddr (contents q)))
(defun quaternion? (q)
  (equal (type-tag q) 'quaternion))

;(defmacro for-each-q-comps (vs f)
;  `(list ,@(for-each-comps-apply '(sigma-1 sigma-x sigma-y sigma-z) vs f)))
(defmacro for-each-q-comps (vs f)
  `(list ,@(mapcar (lambda (comp)
                     `(,f ,@(mapcar (lambda (v) `(,comp ,v)) vs)))
                   '(sigma-1 sigma-x sigma-y sigma-z))))

(defun q-add-num (q1 n)
  (make-quat
    (num-add-cells n (sigma-1 q1))
    (sigma-x q1) (sigma-y q1) (sigma-z q1)))
(defun q-add-cells (q1 cs)
  (make-quat
    (cells-sum `(,(sigma-1 q1) ,cs))
    (sigma-x q1) (sigma-y q1) (sigma-z q1)))
(defun q-add-q (q1 q2)
  (apply #'make-quat
         (for-each-q-comps
           (q1 q2)
           (lambda (x y)
             (cells-sum (list x y))))))
(defun sum-2q (q1 q2)
  (cond ((quaternion? q1)
         (cond ((quaternion? q2) (q-add-q q1 q2))
               ((numberp q2) (q-add-num q1 q2))
               (t (q-add-cells q1 q2))))
        ((numberp q1)
         (cond ((quaternion? q2) (q-add-num q2 q1))
               ((numberp q2)
                (make-quat (make-cell (+ q1 q2) '() '())
                           '() '() '()))
               (t (make-quat (num-add-cells q1 q2)
                             '() '() '()))))
        (t (cond ((quaternion? q2) (q-add-cells q2 q1))
                 ((numberp q2)
                  (make-quat (num-add-cells q2 q1)
                             '() '() '()))
                 (t (make-quat (cells-add-cells q1 q2)
                               '() '() '()))))))

;;; q multiply q
(defun q-multiply-num (q n)
  (apply #'make-quat
         (for-each-q-comps
           (q)
           (lambda (q-comp)
             (cells-multiply-num q-comp n)))))
(defun q-multiply-cells (q cs)
  (apply #'make-quat
         (for-each-q-comps
           (q)
           (lambda (q-comp)
             (cells-multiply-cells q-comp cs)))))
(defun cells-multiply-q (cs q)
  (apply #'make-quat
         (for-each-q-comps
           (q)
           (lambda (q-comp)
             (cells-prod (list cs q-comp)))))) 
;;; Dirac formulation:
(defmacro -with-q-cross- (q1 q2 comp1 comp2 comp3)
    ;; sigma_x = 1 * B_comp1 + A_comp1 * 1 + i (A_comp2 B_comp3 - A_comp3 B_comp2)
    `(cells-sum (list (cells-multiply-cells (sigma-1 ,q1) (,comp1 ,q2))
                      (cells-multiply-cells (,comp1 ,q1) (sigma-1 ,q2))
                      (cells-multiply-num
                        (cells-multiply-cells (,comp2 ,q1) (,comp3 ,q2)) #C(0 1))
                      (cells-multiply-num
                        (cells-multiply-cells (,comp3 ,q1) (,comp2 ,q2)) #C(0 -1)))))
(defun q-multiply-q (q1 q2)
  "\sigma A \sigma B = A \cdot B + i \sigma \cdot A \cross B"
  (make-quat
    ;; 1 = 1 * 1 + A \cdot B
    (cells-sum (for-each-q-comps
                 (q1 q2)
                 (lambda (x y) (cells-prod (list x y)))))
    ;; sigma_x = 1 * B_x + A_x * 1 + i (A_y B_z - A_z B_y)
    (-with-q-cross- q1 q2 sigma-x sigma-y sigma-z)
    ;; sigma_y = 1 * B_y + A_y * 1 + i (A_z B_x - A_x B_z)
    (-with-q-cross- q1 q2 sigma-y sigma-z sigma-x)
    ;; sigma_z = 1 * B_z + A_z * 1 + i (A_x B_y - A_y B_x)
    (-with-q-cross- q1 q2 sigma-z sigma-x sigma-y)))
;;; reduce a string of qs, return a quaternion
;;; e.g. (reduce-qs (list rkb rrb rrb rkb))
(defun contract-2q (q1 q2)
  (cond ((quaternion? q1)
         (cond ((quaternion? q2) (q-multiply-q q1 q2))
               ((numberp q2) (q-multiply-num q1 q2))
               (t (q-multiply-cells q1 q2))))
        ((numberp q1)
         (cond ((quaternion? q2) (q-multiply-num q2 q1))
               ((numberp q2)
                (make-quat (make-cell (* q1 q2) '() '())
                           '() '() '()))
               (t (make-quat (cells-multiply-num q2 q1)
                             '() '() '()))))
        (t (cond ((quaternion? q2) (cells-multiply-q q1 q2))
                 ((numberp q2)
                  (make-quat (cells-multiply-num q1 q2)
                             '() '() '()))
                 (t (make-quat (cells-multiply-cells q1 q2)
                               '() '() '()))))))

(defun reduce-qs (qs)
  (cond ((null qs) qs)
        (t (reduce #'contract-2q qs))))

;;; vector = (comp-x comp-y comp-z)
(defun make-vec (x y z)
  (attach-tag 'vector (list x y z)))
(defun comp-x (vec)
  (car (contents vec)))
(defun comp-y (vec)
  (cadr (contents vec)))
(defun comp-z (vec)
  (caddr (contents vec)))
(defun vector? (v)
  (equal (type-tag v) 'vector))

;(defmacro for-each-vector-comps (vs f)
;  `(list ,@(for-each-comps-apply '(comp-x comp-y comp-z) vs f)))
(defmacro for-each-vector-comps (vs f)
  `(list ,@(mapcar (lambda (comp)
                     `(,f ,@(mapcar (lambda (v) `(,comp ,v)) vs)))
                   '(comp-x comp-y comp-z))))

(defmacro -with-vec-cross- (v1 v2 comp1 comp2)
  `(q-add-q (contract-2q (,comp1 ,v1) (,comp2 ,v2))
            (q-multiply-num (contract-2q (,comp2 ,v1) (,comp1 ,v2)) -1)))
(defun v-cross-v (v1 v2)
  "vec1 \cross vec2"
  (cond ((and (vector? v1) (vector? v2))
         (make-vec (-with-vec-cross- v1 v2 comp-y comp-z)
                   (-with-vec-cross- v1 v2 comp-z comp-x)
                   (-with-vec-cross- v1 v2 comp-x comp-y)))
        (t (error "v1 v2 must be vectors"))))
(defun v-dot-v (v1 v2)
  "dot product"
  (cond ((and (vector? v1) (vector? v2))
         (reduce #'q-add-q
                 (for-each-vector-comps (v1 v2) contract-2q)))
        (t (error "v1 v2 must be vectors"))))

(defun q-multiply-v (q v)
  "scalar multiply vector"
  (apply #'make-vec
         (for-each-vector-comps
           (v)
           (lambda (x) (contract-2q q x)))))
(defun v-multiply-q (v q)
  "vector multiply scalar"
  (apply #'make-vec
         (for-each-vector-comps
           (v)
           (lambda (x) (contract-2q x q)))))

;;; imply reduce-qs, return tensor of quaternions
(defun reduce-vs (vs)
  "reduce vectors to generate tensors"
  (labels ((reduce-vs-iter (v1 v2)
             (cond ((vector? v1)
                    (apply #'make-vec
                           (for-each-vector-comps
                             (v1)
                             (lambda (x) (reduce-vs-iter x v2)))))
                   (t (cond ((vector? v2)
                             (apply #'make-vec
                                    (for-each-vector-comps
                                      (v2)
                                      (lambda (x) (reduce-vs-iter v1 x))))) 
                            (t (contract-2q v1 v2)))))))
    (cond ((null vs) vs)
          (t (reduce #'reduce-vs-iter vs
                     :initial-value (make-cell 1 '() '()))))))

;; vim: ft=lisp
