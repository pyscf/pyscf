!/*
! * Author: Peter Koval <koval.peter@gmail.com>
! */

subroutine hello_world( a, d) bind(c)
  use iso_c_binding, only: C_CHAR, C_NULL_CHAR, C_DOUBLE, C_INT
  implicit none
  !! external
  integer(c_int), intent(in) :: a
  real(c_double), intent(in) :: d
  !! internal

  write(6,*) __FILE__, __LINE__, a, d

end subroutine ! hello_world


