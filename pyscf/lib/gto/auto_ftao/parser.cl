;;;;
;;;; Copyright (C) 2013-  Qiming Sun <osirpt.sun@gmail.com>
;;;; Description:
;;;;    translate the short expression to the expression which is
;;;;    suitable for derivator

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; TODO: the sum of several integrals
; merge these integrals if they
; 1. have the same sequence of operators like r, p, rinv, ...
; 2. have the same rank of tensor
; 3. for each component, have the same types, i.e. cells/quaternions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; parse the expression such as ( op,op |h| op,op )

(load "utility.cl")
(load "derivator.cl")

;;; ovlp
;;; nuc ~ rinv
;;; nabla-rinv = \vec{r}/r^3 = -\nabla{1/r}
;;; rinv  = 1/r
;;; r12   = 1/r_12
;;; grids = a list of 1/r_{grids}
;;; nabla-r12 = \vec{r}_{12}/r_{12}^3 = -\nabla_1{1/r_{12}}
;;; translate these keys in function dress-other combo-op
(defparameter *one-electron-operator* '(ovlp rinv nuc nabla-rinv ccc1e grids))
(defparameter *two-electron-operator* '(r12 ccc2e nabla-r12 gaunt breit-r1 breit-r2))
(defparameter *one-componet-operator* '(rinv nuc grids r12))
(defparameter *nabla-not-comutable* '(rinv nuc grids nabla-rinv r12 nabla-r12 gaunt breit-r1 breit-r2))
(defparameter *act-left-right* '(nabla-rinv nabla-r12 breit-r1 breit-r2))
(defparameter *basic-operators* (append *one-electron-operator* *two-electron-operator*))

;;; *operator*: precedence from high to low
;;; translate these keys in function dress-vec and dress-comp ..
(defparameter *operator* '(vec comp-x comp-y comp-z cross dot))
(defparameter *vec-operator* '(cross dot))
;;; p = -i \nabla
;;; ip = \nabla
;;; r0 = r - (0,0,0)
;;; rc = r - R(env[PTR_COMMON_ORIG])
;;; ri = r - R_i
;;; rj = r - R_j
;;; rk = r - R_k
;;; rl = r - R_l
;;; r = ri/rj/rk/rl; associate with the basis it acts on
;;; g = i/2 (R_m - R_n) cross r0
;;;
;;; sticker symbol *, means the operator should be combined to ket-ops
;;;         (bra-ops ... p* |op| ket-ops)
;;; here p* will be evaluated with op or ket-ops if they
;;; have cross or dot operators. Using the sticker symbol for p/nabla
;;; to prevent p/nabla operators comutating with the next p/nabla operators

;;; rscalar?
(defparameter *intvar* '(p ip nabla px py pz
                         p* ip* nabla* px* py* pz*
                         r r0 rc ri rj rk rl g
                         x x0 xc xi xj xk xl
                         y y0 yc yi yj yk yl
                         z z0 zc zi zj zk zl))
(defparameter *var-nabla* '(p ip nabla px py pz))
(defparameter *var-sticker* '(p* ip* nabla* px* py* pz*))
(defparameter *var-vec* '(p ip nabla p* ip* nabla* r r0 rc ri rj rk rl g))

(defun gaunt-int? (expr)
  (member 'gaunt expr))
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
    ((r ri rj rk rl r0 rc nabla nabla*)
     (make-vec (make-cell 1 '() `(,item x))
               (make-cell 1 '() `(,item y))
               (make-cell 1 '() `(,item z))))
    ((p) (make-vec (make-cell #C(0 -1) '() '(nabla x))
                   (make-cell #C(0 -1) '() '(nabla y))
                   (make-cell #C(0 -1) '() '(nabla z))))
    ((p*) (make-vec (make-cell #C(0 -1) '() '(nabla* x)) ; prevent commutating to p
                    (make-cell #C(0 -1) '() '(nabla* y))
                    (make-cell #C(0 -1) '() '(nabla* z))))
    ((g) ; Rmn cross r0
     (v-cross-v (make-vec (make-cell #C(0 1) '(Rmn x) '())
                          (make-cell #C(0 1) '(Rmn y) '())
                          (make-cell #C(0 1) '(Rmn z) '()))
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
    ((px*) (make-cell #C(0 -1) '() '(nabla* z)))
    ((py*) (make-cell #C(0 -1) '() '(nabla* y)))
    ((pz*) (make-cell #C(0 -1) '() '(nabla* z)))
    ;((ipx nablax*) (make-cell 1 '() '(nabla* x)))
    ;((ipy nablay*) (make-cell 1 '() '(nabla* y)))
    ;((ipz nablaz*) (make-cell 1 '() '(nabla* z)))
    ((nuc rinv grids) ; *one-electron-operator*
     (make-cell 1 '() (make-op item 'S)))
    ((nabla-rinv nabla-r12)
     (make-vec (make-cell 1 '() `(,item x))
               (make-cell 1 '() `(,item y))
               (make-cell 1 '() `(,item z))))
    ((ovlp ccc1e) ; *one-electron-operator*
     (make-cell 1 '() '()))
    ((r12 gaunt ccc2e) ; *two-electron-operator*
     (make-cell 1 '() (make-op 'r12 'S)))
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

(defun int4c2e? (expr)
  (eql (count '\, expr) 2))
(defun int3c2e? (expr)
  (or (member 'ccc2e expr)
      (and (member '\| expr)
           (not (intersection *one-electron-operator* expr))
           (eql (count '\, expr) 1))))
(defun int3c1e? (expr)
  (or (member 'ccc1e expr)
      (and (intersection *one-electron-operator* expr)
           (> (count '\, expr) 0))
      (and (not (member '\| expr))
           (eql (count '\, expr) 2))))
(defun int2c2e? (expr)
  (and (eql (count '\, expr) 0)
       (intersection *two-electron-operator* expr)))
(defun two-electron-int? (expr)
  (or (and (member '\, expr) (member '\| expr))
      (intersection *two-electron-operator* expr)))
(defun one-electron-int? (expr)
  (not (two-electron-int? expr)))
(defun int1e-grids? (expr)
  (member 'grids expr))

; split at key, and the first half part is in reversed order
(defun braket-split-at (key lst)
  (let ((acc nil))
    (do ((src lst (cdr src)))
      ((or (null src) (eql key (car src)))
       (values acc (cdr src)))
      (push (car src) acc))))
(defun split-at (key lst)
  (multiple-value-bind (bra ket) (braket-split-at key lst)
    (values (reverse bra) ket)))

(defun split-at* (key lst)
  (labels ((split-iter (coll lst)
             (multiple-value-bind
               (left right)
               (split-if (lambda (x) (eql x key)) lst)
               (cond ((null right) (cons left coll))
                     ((null (cdr right))
                      `(() ,left ,@coll))
                     (t (split-iter (cons left coll)
                                    (cdr right)))))))
    (reverse (split-iter '() lst))))

(defun first-half-when-split-at (key lst)
  (split-at key lst))
(defun second-half-when-split-at (key lst)
  (nth-value 1 (split-at key lst)))

(defun split-int-expression (expr)
  "return list (op bra-i ket-j bra-k ket-l)"
  (let ((ops (remove-if #'numberp expr)))
    (if (two-electron-int? ops)
      (let ((items (split-at* '\| ops)))
        (if (eql (length items) 2)
            `((r12)
              ,@(multiple-value-list (split-at '\, (car items)))
              ,@(multiple-value-list (split-at '\, (cadr items))))
            (let ((ops (cadr items))
                  (ij (multiple-value-list (split-at '\, (car items))))
                  (kl (multiple-value-list (split-at '\, (caddr items)))))
              (cond ((member 'breit-r1 ops)
; Insert r between ij and between kl to make breit term have correct operator
; sequence (and correct tot-bits).  The operator sequence (tot-bits) are used by
; file gen-code.cl (see i-len, j-len, k-len, l-len) to generate intermediates g0,g1,g2...
                     (list ops (car ij) (cons 'r0 (cadr ij)) (car kl) (cadr kl)))
                    ((member 'breit-r2 ops)
                     (list ops (car ij) (cadr ij) (car kl) (cons 'r0 (cadr kl))))
                    (t `(,ops ,@ij ,@kl))))))
      ; one-electron-int
      (if (int3c1e? ops)
        (let ((items (split-at* '\, ops)))
          (list '(ccc1e) (car items) (cadr items) (caddr items) '()))
        (let ((items (split-at* '\| ops)))
          (if (eql (length items) 2)
            (list '(ovlp) (car items) (cadr items) '() '())
            (list (cadr items) (car items) (caddr items) '() '())))))))
(defun bra-1-of (expr)
  (caddr (split-int-expression expr)))
(defun bra-2-of (expr)
  (caddddr (split-int-expression expr)))
(defun ket-1-of (expr)
  (cadddr (split-int-expression expr)))
(defun ket-2-of (expr)
  (cadddddr (split-int-expression expr)))
(defun operator-of (expr)
  (cadr (split-int-expression expr)))
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

(defun count-tensor-rank (ops)
  (let ((ndot (count 'dot ops)) ; a dot operator reduces two ranks
        (ncross (count 'cross ops)) ; a cross operator reduces one rank
        (nvec (count-if #'(lambda (x) (member x *var-vec*)) ops)))
    (- nvec ncross (* 2 ndot))))

(defun attach-sticker (op)
  (if (member op *var-nabla*)
      (cond ((eql op 'p)     'p*)
            ((eql op 'ip)    'ip*)
            ((eql op 'nabla) 'nabla*)
            ((eql op 'px)    'px*)
            ((eql op 'py)    'py*)
            ((eql op 'pz)    'pz*)
            (t (error (format nil "Unknown var-nabla op ~a%" op))))
      op))

(defun deattach-sticker (op)
  (if (member op *var-sticker*)
      (cond ((eql op 'p*)     'p)
            ((eql op 'ip*)    'ip)
            ((eql op 'nabla*) 'nabla)
            ((eql op 'px*)    'px)
            ((eql op 'py*)    'py)
            ((eql op 'pz*)    'pz)
            (t (error (format nil "Unknown var-sticker op ~a%" op))))
      op))

(defun normal-order-vec-r12 (expr)
  ; change the expression (p dot rinv xxx) to (p rinv dot xxx)
  (cond ((last-one? expr) expr)
        ((and (member (car expr) *operator*)
              (member (cadr expr) *one-componet-operator*))
         (cons (cadr expr) (cons (car expr) (cddr expr))))
        (t (cons (car expr) (normal-order-vec-r12 (cdr expr))))))

(defun remove-sticker-next-op (expr) ; remove the operator behind sticker
  (cond ((last-one? expr) expr)
        ((and (member (car expr) *var-sticker*)
              (member (cadr expr) *nabla-not-comutable*)
              (member (caddr expr) *vec-operator*))
         (if (member (cadr expr) *one-componet-operator*)
             (cons (car expr) (cddr expr))
             (error (format nil "Unsupported pattern ~a ~a%" (car expr) (cadr expr)))))
        ((and (member (car expr) *var-nabla*)
              (member (cadr expr) *nabla-not-comutable*)
              (member (caddr expr) *vec-operator*))
           ; prevent the case p rinv cross p = 0
           (cons (attach-sticker (car expr)) (cddr expr)))
        ((eql (cadr expr) 'ovlp)
         ; ensure the case p ovlp cross p = 0
         (cons (deattach-sticker (car expr)) (cddr expr)))
        (t (cons (car expr) (remove-sticker-next-op (cdr expr))))))

(defun dagger-append (bra ket)
  (cond ((null bra) ket)
        (t (dagger-append (cdr bra)
                          (cons (car bra) ket)))))

(defun cons-bra-ket (bra ket &optional op)
  ; when bra is reversed, every 'cross gives -1 and p^* maps to -p
  (let ((fac (* (expt -1 (count-if (lambda (x)
                                     (member x '(p p* px py pz px* py* pz*)))
                                   bra))
                (expt -1 (count 'cross bra))))
        (terms (remove-sticker-next-op
                 (normal-order-vec-r12
                   (dagger-append bra (append op ket))))))
      (cons fac terms)))

(defun reverse-tensor-ordering (ts &optional (depth 0))
  (labels ((leafpend (xnode ynode znode depth)
             (if (and (> depth 0) (listp xnode) (vector? xnode))
                 (make-vec (leafpend (comp-x xnode) (comp-x ynode) (comp-x znode) (1- depth))
                           (leafpend (comp-y xnode) (comp-y ynode) (comp-y znode) (1- depth))
                           (leafpend (comp-z xnode) (comp-z ynode) (comp-z znode) (1- depth)))
                 (make-vec xnode ynode znode))))
    (if (and (> depth 0) (listp ts) (vector? ts))
        (leafpend (reverse-tensor-ordering (comp-x ts) (1- depth))
                  (reverse-tensor-ordering (comp-y ts) (1- depth))
                  (reverse-tensor-ordering (comp-z ts) (1- depth))
                  depth)
        ts)))

(defun format-vs-1e (phase ops-i ops-j &optional op)
  (let* ((ops (cons phase (cons-bra-ket ops-i ops-j op)))
         (rankbra (count-tensor-rank ops-i))
         (vs (reverse-tensor-ordering
              (reduce-vs (reduce-rpn (infix-to-rpn ops)))
              (1- rankbra)))
         (ts (label-ts vs))
         (sf (label-sf vs))
         (p-vs (if (eql ts 'ts)
                 (map-q-in-vs #'plain-ts vs)
                 (map-q-in-vs #'plain-tas vs))))
    (list ts sf (if (eql sf 'sf)
                  (map-q-in-vs #'filter-ifnot-sf p-vs)
                  p-vs))))
(defun plainq-by-plainq (q1 q2)
  "product of two plain quaternions, do not squeeze. ordered like
  ((1_x 2_x) (1_y 2_x) (1_z 2_x) ...)"
  (cond ((or (null q1) (null q2)) '())
        ((and (or (cell? q1) (cells? q1))
              (or (cell? q2) (cells? q2)))
         (cells-multiply-cells q1 q2))
        ((or (cell? q2) (cells? q2))
         (mapcar (lambda (cs) (cells-multiply-cells cs q2)) q1))
        ((or (cell? q1) (cells? q1))
         (mapcar (lambda (cs) (cells-multiply-cells q1 cs)) q2))
        (t (append (mapcar (lambda (cs)
                             (cells-multiply-cells cs (car q2)))
                           q1)
                   (plainq-by-plainq q1 (cdr q2))))))
(defun vs-by-vs (vs1 vs2)
  "product of two tensors/vectors, do not squeeze. ordered like
  ((1_x 2_x) (1_x 2_y) (1_x 2_z) (1_y 2_x) (1_y 2_y) ...)"
  (map-q-in-vs (lambda (q1)
                 (map-q-in-vs (lambda (q2)
                                (plainq-by-plainq q1 q2))
                              vs2))
               vs1))

(defun eval-int-r12 (phasefac op bra-i ket-j bra-k ket-l)
  (let* ((vs1 (format-vs-1e phasefac bra-i ket-j op))
         (ts1 (car vs1))
         (sf1 (cadr vs1))
         (pv1 (caddr vs1))
         (vs2 (if (gaunt-int? op)
                (format-vs-1e 1 bra-k ket-l op)
                (format-vs-1e 1 bra-k ket-l '(r12))))
         (ts2 (car vs2))
         (sf2 (cadr vs2))
         (pv2 (caddr vs2)))
    (list ts1 sf1 ts2 sf2
          (vs-by-vs pv1 pv2))))

;;; [\sigma_1 \dot (-\nabla 1/r12)] (\sigma_2 \dot r1)
(defun eval-breit-gauge-r1 (phasefac bra-i ket-j bra-k ket-l)
  (flet ((breit-iter (comp)
; (cdr ket-j) to remove first operator because it was added in
; split-int-expression function to get the right tot-btis
           (let* ((vs1 (format-vs-1e phasefac bra-i (cons `(,comp r0) (cdr ket-j))
                                     '(sigma dot nabla-r12)))
                  (ts1 (car vs1))
                  (sf1 (cadr vs1))
                  (pv1 (caddr vs1))
                  (vs2 (format-vs-1e 1 bra-k ket-l `(,comp sigma r12)))
                  (ts2 (car vs2))
                  (sf2 (cadr vs2))
                  (pv2 (caddr vs2)))
             (list ts1 sf1 ts2 sf2 (vs-by-vs pv1 pv2)))))
    (let ((sx (breit-iter 'comp-x))
          (sy (breit-iter 'comp-y))
          (sz (breit-iter 'comp-z)))
      `(,@(subseq sx 0 4)
         ,(reduce #'vs-align-merge (mapcar #'last1 (list sx sy sz)))))))

;;; [\sigma_1 \dot (-\nabla 1/r12)] (\sigma_2 \dot r2)
(defun eval-breit-gauge-r2 (phasefac bra-i ket-j bra-k ket-l)
  (let* ((vs1 (format-vs-1e phasefac bra-i ket-j '(sigma dot nabla-r12)))
         (ts1 (car vs1))
         (sf1 (cadr vs1))
         (pv1 (caddr vs1))
; (cdr ket-l) to remove first operator because it was added in
; split-int-expression function to get the right tot-btis
         (vs2 (format-vs-1e 1 bra-k (append '(sigma dot r0) (cdr ket-l)) '(r12)))
         (ts2 (car vs2))
         (sf2 (cadr vs2))
         (pv2 (caddr vs2)))
    (list ts1 sf1 ts2 sf2 (vs-by-vs pv1 pv2))))

(defun vs-align-merge (vs1 vs2)
  "sum of two tensors/vectors, vs1 and vs2 should be aligned"
  (cond ((null vs1) '())
        ((vector? vs1)
         (apply #'make-vec
                (for-each-vector-comps (vs1 vs2) vs-align-merge)))
        ((or (cell? vs1) (cells? vs1))
         (cells-add-cells vs1 vs2))
        (t (mapcar #'cells-add-cells vs1 vs2)))) ; quaternions
(defun eval-int-gaunt (phasefac op bra-i ket-j bra-k ket-l)
  (flet ((eval-s (c)
           (eval-int-r12 phasefac `(,c sigma ,@op)
                         bra-i ket-j bra-k ket-l)))
    (let ((sx (eval-s 'comp-x))
          (sy (eval-s 'comp-y))
          (sz (eval-s 'comp-z)))
      `(,@(subseq sx 0 4)
         ,(vs-align-merge (last1 sx)
                          (vs-align-merge (last1 sy) (last1 sz)))))))

(defun eval-int (expr)
  (destructuring-bind (op bra-i ket-j bra-k ket-l)
    (split-int-expression expr)
    (let ((unknow (remove-if (lambda (x)
                               (or (numberp x)
                                   (member x '(sigma))
                                   (member x *intvar*) ; ??
                                   (member x *operator*)
                                   (member x *one-electron-operator*)
                                   (member x *two-electron-operator*)))
                             op)))
      (if (> (length unknow) 0)
        (format t "//Warning: unknown operators: ~a" unknow)))
    (let ((unknow (remove-if (lambda (x)
                               (or (numberp x)
                                   (member x '(sigma))
                                   (member x *intvar*)
                                   (member x *operator*)))
                             `(,@bra-i ,@ket-j ,@bra-k ,@ket-l))))
      (if (> (length unknow) 0)
        (format t "//Warning: unknown keys ~a" unknow)))
    (let ((phasefac (nth-value 1 (factor-of expr)))
          (ops-i (subst 'ri 'r bra-i))
          (ops-j (subst 'rj 'r ket-j))
          (ops-k (subst 'rk 'r bra-k))
          (ops-l (subst 'rl 'r ket-l)))
      (cond ((int3c1e? expr)
             (cond ((intersection '(sigma g) ops-k)
                    (error "sigma, g not allowed in int3c1e expression~%"))
                   (ops-l (error "auxiliary basis should not have ops-l~%"))
                   (t (eval-int-r12 phasefac op ops-i ops-j ops-k '()))))
            ((one-electron-int? expr)
             (format-vs-1e phasefac ops-i ops-j op))
; two-electron integrals
            ((int4c2e? expr)
             (cond ((member 'gaunt op)
                    (eval-int-gaunt phasefac op ops-i ops-j ops-k ops-l))
                   ((member 'breit-r1 op)
                    (eval-breit-gauge-r1 phasefac ops-i ops-j ops-k ops-l))
                   ((member 'breit-r2 op)
                    (eval-breit-gauge-r2 phasefac ops-i ops-j ops-k ops-l))
                   ((or (member 'r12 op) (member 'nabla-r12 op))
                    (eval-int-r12 phasefac op ops-i ops-j ops-k ops-l))
                   (t (error "unsupport int4c2e operator ~s~%" op))))
            ((int3c2e? expr)
             (cond ((member 'gaunt op)
                    (error "3c2e-gaunt not implemented~%"))
                   ((member 'r12 op)
                    (cond ((intersection '(sigma g) ops-k)
                           (error "sigma, g not allowed in int3c2e electron 2 expression~%"))
                          (ops-l (error "auxiliary basis should be in electron 2~%"))
                          (t (eval-int-r12 phasefac op ops-i ops-j ops-k '()))))
                   (t (error "unsupport int3c2e operator ~s~%" op))))
            ((int2c2e? expr)
             (cond ((member 'gaunt op)
                    (error "2c2e-gaunt not implemented~%"))
                   ((member 'r12 op)
                    (cond ((or (intersection '(sigma g) ops-k)
                               (intersection '(sigma g) ops-i))
                           (error "sigma, g not allowed in int2c2e expression~%"))
                          (t (eval-int-r12 phasefac op ops-i '() ops-k '()))))
                   (t (error "unsupport int2c2e operator ~s~%" op))))
            (t (error "unknown integral expression~%"))))))


;; vim: ft=lisp
