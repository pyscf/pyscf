module m_blas_wrapper
  contains

!
!
!
subroutine get_upper_triangulat_matrix(A, Ap, n) bind(c, name="get_upper_triangulat_matrix")
  use iso_c_binding

  integer(C_INT), intent(in), value :: n
  real(C_FLOAT), intent(in) :: A(n, n)
  real(C_FLOAT), intent(out) :: Ap(n*(n+1)/2)

  integer :: i, j, counter

  Ap = 0.0
  counter = 1

  do j=1, N
    do i=1, N
      if (j >= i) then
        Ap(counter) = A(i, j)
        counter = counter + 1
      endif
    enddo
  enddo

end subroutine !get_upper_triangulat_matrix

!
!
!
subroutine SSPMV_wrapper(uplo, n, alpha, ap, x, incx, beta, y, incy) bind(c, name="SSPMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: uplo, n, incx, incy
  real(C_FLOAT), intent(in), value :: alpha, beta
  real(C_FLOAT), intent(in) :: ap(n*(n+1)/2), x(n)
  real(C_FLOAT), intent(out) :: y(n)

  character(1) :: uplo_ch

  if (uplo == 0) then
    uplo_ch = "U"
  else if (uplo == 1) then
    uplo_ch = "L"
  else
    stop "uplo not 0 or 1!"
  endif

  call sspmv (uplo_ch, n, alpha, ap, x, incx, beta, y, incy);

end subroutine !SSPMV_wrapper

!
!
!
subroutine DSPMV_wrapper(uplo, n, alpha, ap, x, incx, beta, y, incy) bind(c, name="DSPMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: uplo, n, incx, incy
  real(C_DOUBLE), intent(in), value :: alpha, beta
  real(C_DOUBLE), intent(in) :: ap(n*(n+1)/2), x(n)
  real(C_DOUBLE), intent(out) :: y(n)

  character(1) :: uplo_ch


  if (uplo == 0) then
    uplo_ch = "U"
  else if (uplo == 1) then
    uplo_ch = "L"
  else
    stop "uplo not 0 or 1!"
  endif

  call dspmv (uplo_ch, n, alpha, ap, x, incx, beta, y, incy);

end subroutine !SSPMV_wrapper

!
!
!
subroutine SGEMV_wrapper(trans, m, n, alpha, ap, lda, x, incx, beta, y, incy) bind(c, name="SGEMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: trans, m, n, lda, incx, incy
  real(C_FLOAT), intent(in), value :: alpha, beta
  real(C_FLOAT), intent(in) :: ap(n, n), x(n)
  real(C_FLOAT), intent(out) :: y(n)

  character(1) :: trans_ch

  if (trans == 0) then
    trans_ch = "N"
  else if (trans == 1) then
    trans_ch = "L"
  else
    stop "uplo not 0 or 1!"
  endif

  call sgemv (trans_ch, m, n, alpha, ap, lda, x, incx, beta, y, incy);

end subroutine !SSPMV_wrapper


!
!
!
subroutine DGEMV_wrapper(trans, m, n, alpha, ap, lda, x, incx, beta, y, incy) bind(c, name="DGEMV_wrapper")
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: trans, m, n, lda, incx, incy
  real(C_DOUBLE), intent(in), value :: alpha, beta
  real(C_DOUBLE), intent(in) :: ap(n, n), x(n)
  real(C_DOUBLE), intent(out) :: y(n)

  character(1) :: trans_ch

  if (trans == 0) then
    trans_ch = "N"
  else if (trans == 1) then
    trans_ch = "L"
  else
    stop "uplo not 0 or 1!"
  endif

  call dgemv (trans_ch, m, n, alpha, ap, lda, x, incx, beta, y, incy);

end subroutine !SSPMV_wrapper


end module !m_blas_wrapper.f90
