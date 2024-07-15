;;;;
;;;; Copyright (C) 2014-2017  Qiming Sun <osirpt.sun@gmail.com>
;;;;

;;; parse the expression such as ( op,op |h| op,op )

(load "utility.cl")
(load "derivator.cl")

;;; *operator*: precedence from high to low
;;; translate these keys in function dress-vec and dress-comp ..
(defparameter *operator* '(vec comp-x comp-y comp-z cross dot))
;;; p = -i \nabla
;;; ip = \nabla
;;; r0 = r - (0,0,0)
;;; rc = r - R(env[PTR_COMMON_ORIG])
;;; ri = r - R_i
;;; rj = r - R_j
;;; rk = r - R_k
;;; rl = r - R_l
;;; r = ri/rj/rk/rl; associate with the basis it acts on
;;; g = -i R_m cross r0
;;;
;;; sticker symbol *, which sticks the decorated operator to op or ket-ops
;;;         (bra-ops ... p* |op| ket-ops)
;;; the p* in the bra (| will be evaluated with |op| or |ket-ops) (if they
;;; have cross or dot operators). Using the sticker symbol for p/nabla
;;; to prevent p/nabla operators commutating with the next p/nabla operators

;;; rscalar?
(defparameter *intvar* '(p ip nabla px py pz
                         p* ip* nabla* px* py* pz*
                         r r0 rc ri g
                         x x0 xc xi
                         y y0 yc yi
                         z z0 zc zi))
(defparameter *var-vec* '(p ip nabla p* ip* nabla* r r0 rc ri rj rk rl g))

(defun breit-int? (expr)
  (or (member 'breit-r1 expr) (member 'breit-r2 expr)))
;;;;;; convert to reversed polish notation ;;;;;;;;;;;
(defun complex? (n)
  (not (zerop (imagpart n))))

(defun unary-op? (o)
  (or (eql o 'vec)
      (eql o 'comp-x)
      (eql o 'comp-y)
      (eql o 'comp-z)))
(defun binary-op? (o)
  (or (eql o 'cross)
      (eql o 'dot)))

(defun parenthesis (&rest tokens)
  tokens)
(defun pre-unary (o seq)
  (cond ((null seq) seq)
        ((atom seq) seq)
        ((eql (car seq) o)
         (cons (parenthesis (car seq)
                            (pre-unary o (cadr seq)))
               (pre-unary o (cddr seq))))
        (t (cons (pre-unary o (car seq))
                 (pre-unary o (cdr seq))))))
(defun pre-binary (o-pre o seq)
  (cond ((null seq) (list o-pre))
        ((atom seq) seq)
        ((eql (car seq) o)
         (pre-binary (parenthesis o-pre
                                  (car seq)
                                  (pre-binary '() o (cadr seq)))
                     o (cddr seq)))
        ((null o-pre)
         (pre-binary (pre-binary '() o (car seq))
                     o (cdr seq)))
        (t (cons o-pre
                 (pre-binary (pre-binary '() o (car seq))
                             o (cdr seq))))))
(defun precede (seq o)
  "use () to increase the priority of operator o"
  (cond ((unary-op? o) (pre-unary o seq))
        ((binary-op? o) (pre-binary '() o seq))
        (t (error "unknown operator ~a~%" o))))
(defun infix-to-rpn (tokens)
  (labels ((rpn-iter (rpn-stack seq)
             ;; return a rpn stack
             (cond ((null seq) rpn-stack)
                   ((atom seq) (cons seq rpn-stack))
                   ((member (car seq) *operator*)
                    (rpn-iter (cons (car seq)
                                    (rpn-iter rpn-stack (cadr seq)))
                              (cddr seq)))
                   (t (rpn-iter (rpn-iter rpn-stack (car seq)) (cdr seq))))))
    (reverse (rpn-iter '() (reduce #'precede *operator*
                                   :initial-value tokens)))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;; convert infix expression to set of vectors
(defun dress-other (item)
  "based on the symbol of item, return a cell, a quaternion or a vector"
  (case item
    ((sigma) ; \vec{\sigma}
     (make-vec (make-quat '(0 ()) '(1 ()) '(0 ()) '(0 ()))
               (make-quat '(0 ()) '(0 ()) '(1 ()) '(0 ()))
               (make-quat '(0 ()) '(0 ()) '(0 ()) '(1 ()))))
    ((r ri rj rk rl r0 rc nabla)
     (make-vec (make-cell 1 '() `(,item x))
               (make-cell 1 '() `(,item y))
               (make-cell 1 '() `(,item z))))
    ((p) (make-vec (make-cell #C(0 -1) '() '(nabla x))
                   (make-cell #C(0 -1) '() '(nabla y))
                   (make-cell #C(0 -1) '() '(nabla z))))
    ((g) ; -R_m cross r0
     (v-cross-v (make-vec (make-cell #C(0 1) '(Rm x) '())
                          (make-cell #C(0 1) '(Rm y) '())
                          (make-cell #C(0 1) '(Rm z) '()))
                (make-vec (make-cell 1 '() '(r0 x))
                          (make-cell 1 '() '(r0 y))
                          (make-cell 1 '() '(r0 z)))))
    ((x y z) (make-cell 1 '() (make-op 'r item)))
    ((x0) (make-cell 1 '() (make-op 'r0 'x)))
    ((y0) (make-cell 1 '() (make-op 'r0 'y)))
    ((z0) (make-cell 1 '() (make-op 'r0 'z)))
    ((xc) (make-cell 1 '() (make-op 'rc 'x)))
    ((yc) (make-cell 1 '() (make-op 'rc 'y)))
    ((zc) (make-cell 1 '() (make-op 'rc 'z)))
    ((xi) (make-cell 1 '() (make-op 'ri 'x)))
    ((yi) (make-cell 1 '() (make-op 'ri 'y)))
    ((zi) (make-cell 1 '() (make-op 'ri 'z)))
    ((xj) (make-cell 1 '() (make-op 'rj 'x)))
    ((yj) (make-cell 1 '() (make-op 'rj 'y)))
    ((zj) (make-cell 1 '() (make-op 'rj 'z)))
    ((xk) (make-cell 1 '() (make-op 'rk 'x)))
    ((yk) (make-cell 1 '() (make-op 'rk 'y)))
    ((zk) (make-cell 1 '() (make-op 'rk 'z)))
    ((xl) (make-cell 1 '() (make-op 'rl 'x)))
    ((yl) (make-cell 1 '() (make-op 'rl 'y)))
    ((zl) (make-cell 1 '() (make-op 'rl 'z)))
    ((px) (make-cell #C(0 -1) '() '(nabla z)))
    ((py) (make-cell #C(0 -1) '() '(nabla y)))
    ((pz) (make-cell #C(0 -1) '() '(nabla z)))
    ;((ipx nablax) (make-cell 1 '() '(nabla x)))
    ;((ipy nablay) (make-cell 1 '() '(nabla y)))
    ;((ipz nablaz) (make-cell 1 '() '(nabla z)))
    (otherwise (if (numberp item)
                 (make-cell item '() '()) ; factor
                 (make-cell 1 (make-op item 'S) '()))))) ; constant; 'S indicates a scalar
(defun dress-comp (comp item)
  (cond ((vector? item) (funcall comp item))
        ((cell? item)
         (let* ((n (phase-of item))
                (const (consts-of item))
                (op (ops-of item))
                (script (funcall comp (make-vec 'x 'y 'z))))
           (cond ((scalar? op)
                  (make-cell n const (make-op (symbol-of op) script)))
                 ((scalar? const)
                  (make-cell n (make-op (symbol-of const) script) op))
                 (t item))))
        (t item)))
(defun dress-vec (item)
  (cond ((vector? item) item)
        ((quaternion? item)
         (make-vec item item item))
        ((cell? item)
         (let* ((n (phase-of item))
                (const (consts-of item))
                (op (ops-of item)))
           (cond ((scalar? op) ; extend scalar to vector
                  (make-vec (make-cell n const (make-op (symbol-of op) 'x))
                            (make-cell n const (make-op (symbol-of op) 'y))
                            (make-cell n const (make-op (symbol-of op) 'z))))
                 ((scalar? const)
                  (make-vec (make-cell n (make-op (symbol-of const) 'x) op)
                            (make-cell n (make-op (symbol-of const) 'y) op)
                            (make-cell n (make-op (symbol-of const) 'z) op)))
                 (t item))))
        (t (error "unknown type ~a~%" item))))

(defun reduce-rpn (rpn)
  "reduce the reversed polish notation to a set of vectors"
  (flet ((reduce-rpn-iter (stack token)
           (case token
             (('()) stack)
             ((vec) (cons (dress-vec (car stack))
                          (cdr stack)))
             ((dot) (cons (v-dot-v (cadr stack) (car stack))
                          (cddr stack)))
             ((cross) (cons (v-cross-v (cadr stack) (car stack))
                            (cddr stack)))
             ((comp-x comp-y comp-z)
              ; place comp-* after dot/cross to extract one
              ; component of dot/cross production
              (cons (dress-comp token (car stack))
                    (cdr stack)))
             (otherwise (cons (dress-other token) stack)))))
    (reverse (reduce #'reduce-rpn-iter rpn
               :initial-value '()))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun q?-in-vs (vsfunc qfunc vs)
  "query and apply func on each q in vs"
  (cond ((vector? vs)
         (apply vsfunc
                (for-each-vector-comps
                  (vs)
                  (lambda (x) (q?-in-vs vsfunc qfunc x)))))
        (t (funcall qfunc vs)))) ;either cellss or quaternion or nil

(defun query-q-in-vs (func vs)
  (flet ((f-and (&rest args) (every #'identity args)))
    (q?-in-vs #'f-and func vs)))
(defun cells-complex? (cs)
  (or (null cs)
      (cond ((cell? cs) (complex? (phase-of cs)))
            (t (complex? (phase-of (car cs)))))))
(defun cells-real? (cs)
  (or (null cs)
      (cond ((cell? cs) (zerop (imagpart (phase-of cs))))
            (t (zerop (imagpart (phase-of (car cs))))))))
(defun ts? (q)
  (cond ((null q) t)
        ((or (cell? q) (cells? q))
         (cells-real? (car q)))
        ((quaternion? q)
         (and (cells-real?    (sigma-1 q))
              (cells-complex? (sigma-x q))
              (cells-complex? (sigma-y q))
              (cells-complex? (sigma-z q))))))
(defun tas? (q)
  (cond ((null q) t)
        ((or (cell? q) (cells? q))
         (cells-complex? (car q)))
        ((quaternion? q)
         (and (cells-complex? (sigma-1 q))
              (cells-real?    (sigma-x q))
              (cells-real?    (sigma-y q))
              (cells-real?    (sigma-z q))))))
(defun label-ts (vs)
  (cond ((query-q-in-vs #'ts? vs) 'ts)
        ((query-q-in-vs #'tas? vs) 'tas)
        (t (error "neither ts nor tas~%"))))

(defun spin-free? (vs)
  (cond ((null vs) t)
        ((vector? vs)
         (and (spin-free? (comp-x vs))
              (spin-free? (comp-y vs))
              (spin-free? (comp-z vs))))
        ((or (cell? vs) (cells? vs)) t)
        (t (and (null (sigma-x vs))
                (null (sigma-y vs))
                (null (sigma-z vs))))))
(defun label-sf (vs)
  (if (query-q-in-vs #'spin-free? vs)
    'sf
    'si)) ; spin included

(defun map-q-in-vs (func vs)
  (q?-in-vs #'make-vec func vs))
;;;ts  = [1 is_x is_y is_z], dump [s_x s_y s_z 1]
;;;tas = [i s_x s_y s_z] = [1, -is_x -is_y -is_z] * i, dump [-s_x -s_y -s_z 1]
(defun plain-ts (q)
  (list (cells-multiply-num (sigma-x q) #C(0 -1))
        (cells-multiply-num (sigma-y q) #C(0 -1))
        (cells-multiply-num (sigma-z q) #C(0 -1))
        (sigma-1 q)))
(defun plain-tas (q)
  (list (cells-multiply-num (sigma-x q) -1)
        (cells-multiply-num (sigma-y q) -1)
        (cells-multiply-num (sigma-z q) -1)
        (cells-multiply-num (sigma-1 q) #C(0 -1))))
(defun filter-ifnot-sf (plain-q)
  (cadddr plain-q))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun factor-of (raw-infix)
  "return realpart and phase"
  (let ((fac (* (apply #'* (remove-if-not #'numberp raw-infix))
                (expt .5 (count 'g raw-infix)))))
    (cond ((zerop (imagpart fac))
           (values fac 1))
          ((zerop (realpart fac))
           (values (imagpart fac) #C(0 1)))
          (t (error "cannot handle complex factor ~a~%" fac)))))

(defun format-vs (phase expr)
  (let* ((ops (cons phase expr))
         (vs (reduce-vs (reduce-rpn (infix-to-rpn ops))))
         (ts (label-ts vs))
         (sf (label-sf vs))
         (p-vs (if (eql ts 'ts)
                 (map-q-in-vs #'plain-ts vs)
                 (map-q-in-vs #'plain-tas vs))))
    (list ts sf (if (eql sf 'sf)
                  (map-q-in-vs #'filter-ifnot-sf p-vs)
                  p-vs))))

(defun eval-gto (expr)
  (let ((unknow (remove-if (lambda (x)
                             (or (numberp x)
                                 (member x '(sigma))
                                 (member x *operator*)
                                 (member x *intvar*)))
                           expr)))
    (if (> (length unknow) 0)
      (format t "//Warning: unknown operators: ~a" unknow)))
    (let ((phasefac (nth-value 1 (factor-of expr))))
      (format-vs phasefac expr)))


;; vim: ft=lisp
