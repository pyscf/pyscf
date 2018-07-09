module sparse_blas_wrapper
  contains

!subroutine SCSRMM_wrapper(trans, m, n, k, alpha, matdescra, val, size_val, indx, 
!  pntrb, pntre, B, dimb1, dimb2, ldb, beta, C, dimc1, dimc2, ldc) bind(c, name="SCSRMM_wrapper")
!  use iso_c_binding
!
!  implicit none
!  integer(C_INT), intent(in), value :: trans, m, n, k, ldb, ldc, size_val, dimb1, dimb2, dimc1, dimc2
!  integer(C_INT), intent(in) :: indx, pntrb, pntre
!  real(C_FLOAT), intent(in), value :: alpha, beta
!
!  real(C_FLOAT), intent(in) :: val(size_val), B(dimb1, dimb2)
!  real(C_FLOAT), intent(out) :: C(dimc1, dimc2)
!
!  character(1) :: trans_ch
!
!  if (trans == 0) then
!    trans_ch = "N"
!  else if (trans == 1) then
!    trans_ch = "T"
!  else if (trans == 2) then
!    trans_ch = "C"
!  else
!    stop "uplo not 0 or 1!"
!  endif
!
!  call mkl_scsrmm (trans_ch, m, n, k, ia, ja, x, y);
!
!
!
!end subroutine ! SCSRMM_wrapper

!
!
!
subroutine SCSRGEMV_wrapper(trans, m, a, size_a, ia, size_ia, ja, size_ja, x, y) bind(c, name="SCSRGEMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: trans, m, size_a, size_ia, size_ja
  integer(C_INT), intent(in) :: ia(size_ia), ja(size_ja)
  real(C_FLOAT), intent(in) :: a(size_a), x(m)
  real(C_FLOAT), intent(out) :: y(m)

  character(1) :: trans_ch

  if (trans == 0) then
    trans_ch = "N"
  else if (trans == 1) then
    trans_ch = "T"
  else
    stop "uplo not 0 or 1!"
  endif

  call mkl_cspblas_scsrgemv (trans_ch, m, a, ia, ja, x, y);

end subroutine !CSRSSPMV_wrapper

!
!
!
subroutine DCSRGEMV_wrapper(trans, m, a, size_a, ia, size_ia, ja, size_ja, x, y) bind(c, name="DCSRGEMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: trans, m, size_a, size_ia, size_ja
  integer(C_INT), intent(in) :: ia(size_ia), ja(size_ja)
  real(C_DOUBLE), intent(in) :: a(size_a), x(m)
  real(C_DOUBLE), intent(out) :: y(m)

  character(1) :: trans_ch

  if (trans == 0) then
    trans_ch = "N"
  else if (trans == 1) then
    trans_ch = "T"
  else
    stop "uplo not 0 or 1!"
  endif

  call mkl_cspblas_dcsrgemv (trans_ch, m, a, ia, ja, x, y);

end subroutine !CSRSSPMV_wrapper

!
!
!
subroutine CCSRGEMV_wrapper(trans, m, a, size_a, ia, size_ia, ja, size_ja, x, y) bind(c, name="CCSRGEMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: trans, m, size_a, size_ia, size_ja
  integer(C_INT), intent(in) :: ia(size_ia), ja(size_ja)
  real(C_FLOAT_COMPLEX), intent(in) :: a(size_a), x(m)
  real(C_FLOAT_COMPLEX), intent(out) :: y(m)

  character(1) :: trans_ch

  if (trans == 0) then
    trans_ch = "N"
  else if (trans == 1) then
    trans_ch = "T"
  else
    stop "uplo not 0 or 1!"
  endif

  call mkl_cspblas_ccsrgemv (trans_ch, m, a, ia, ja, x, y);

end subroutine !CSRSSPMV_wrapper

!
!
!
subroutine ZCSRGEMV_wrapper(trans, m, a, size_a, ia, size_ia, ja, size_ja, x, y) bind(c, name="ZCSRGEMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: trans, m, size_a, size_ia, size_ja
  integer(C_INT), intent(in) :: ia(size_ia), ja(size_ja)
  real(C_DOUBLE_COMPLEX), intent(in) :: a(size_a), x(m)
  real(C_DOUBLE_COMPLEX), intent(out) :: y(m)

  character(1) :: trans_ch

  if (trans == 0) then
    trans_ch = "N"
  else if (trans == 1) then
    trans_ch = "T"
  else
    stop "uplo not 0 or 1!"
  endif

  call mkl_cspblas_zcsrgemv (trans_ch, m, a, ia, ja, x, y);

end subroutine !CSRSSPMV_wrapper

end module sparse_blas_wrapper
