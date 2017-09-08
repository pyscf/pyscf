module m_warn
  implicit none
  
  contains

!
! A warning subroutine
!
subroutine warn(message, line_num)
  use m_color, only : bc
  !! external
  character(*), intent(in), optional :: message
  integer, intent(in), optional :: line_num

  if(present(message)) then
    if(present(line_num)) then
      write(6,'(a,a,a,i7,a)') bc%WARNING, trim(message), ' at line ',line_num,bc%ENDC    
      write(0,'(a,a,a,i7,a)') bc%WARNING, trim(message), ' at line ',line_num,bc%ENDC
    else
      write(6,'(a,a,a)') bc%WARNING, trim(message), bc%ENDC    
      write(0,'(a,a,a)') bc%WARNING, trim(message), bc%ENDC
    endif  
  endif

end subroutine ! warn

end module !m_warn

