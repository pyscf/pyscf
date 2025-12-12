;;;;
;;;; Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
;;;; Description:

(defun last-one? (lst)
  (null (cdr lst)))
(defun last-two? (lst)
  (null (cddr lst)))

(defmacro caddddr (l)
  `(nth 4 ,l))
(defmacro cadddddr (l)
  `(nth 5 ,l))

(defun last1 (lst)
  (car (last lst)))

(defun split-if (fn lst)
  (let ((acc nil))
    (do ((src lst (cdr src)))
      ((or (null src) (funcall fn (car src)))
       (values (nreverse acc) src))
      (push (car src) acc))))


(defun mapa-b (fn a b &optional (step 1))
  (do ((i a (+ i step))
       (result nil))
      ((> i b) (nreverse result))
    (push (funcall fn i) result)))

(defun range (n &optional m)
  (cond ((null m)
         (mapa-b #'1+ -1 (- n 2)))
        (t (mapa-b #'1+ (1- n) (- m 2)))))

;;; > (mkstr pi " pieces of " 'pi)
;;; "3.141592653589793 pieces of PI"
(defun mkstr (&rest args)
  (with-output-to-string (s)
    (dolist (a args) (princ a s))))


(defun for-each-comps-apply (comps vs f)
  (mapcar (lambda (comp)
              `(,f ,(mapcar (lambda (v) `(,comp ,v)) vs)))
            comps))
