module m_conv_dims

  contains

!
!
!
subroutine conv_dims(n1, n2, n12, n122)
  implicit none
  integer, intent(in)  :: n1, n2
  integer, intent(out) :: n12
  integer, intent(out), optional :: n122
  
  n12 = n1+n2
  if(present(n122)) n122 = n12/2+1
  
end subroutine ! conv_dims


end module ! m_conv_dims
