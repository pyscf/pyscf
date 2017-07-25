module m_get_sizeof

  implicit none

  interface get_sizeof
!    module procedure i1
    module procedure i2
    module procedure i4
    module procedure i8
    module procedure r4
    module procedure r8
  end interface ! get_sizeof

  contains

!
! Finds size of integer in bytes
!
!integer function i1(v); integer(1) :: v; i1 = 1; end function 

integer function i2(v); integer(2) :: v; i2 = 2; end function

integer function i4(v); integer(4) :: v; i4 = 4; end function

integer function i8(v); integer(8) :: v; i8 = 8; end function

integer function r4(v); real(4) :: v; r4 = 4; end function

integer function r8(v); real(8) :: v; r8 = 8; end function


end module !m_get_sizeof 

