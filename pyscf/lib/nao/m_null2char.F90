module m_null2char
  use iso_c_binding, only: c_char
  
  implicit none

  contains  

!
!
!
subroutine null2char(c_null, c_fstring)
  implicit none
  !! external 
  character(kind=c_char), intent(in) :: c_null(*)
  character(*), intent(inout) :: c_fstring
  !! internal
  integer :: i, c
  
  c_fstring = ""  
  do i=1,len(c_fstring)
    c = ichar(c_null(i))
    if(c==0) exit
    c_fstring(i:i) = char(c)
  enddo

end subroutine ! 

end module !m_null2char
