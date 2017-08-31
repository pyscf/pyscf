module m_die
  
  implicit none

  contains

!
! The die subroutine
!
subroutine die(message, line_num)
  use m_color, only : bc
  !! external
  character(*), intent(in), optional :: message
  integer, intent(in), optional :: line_num

  !! internal
#ifdef _WIN32
  integer :: ios
#endif

  if(present(message)) then
    if(present(line_num)) then
      write(6,'(a,a,a,i7,a)') bc%FAIL, trim(message), ' at line ',line_num,bc%ENDC    
      write(0,'(a,a,a,i7,a)') bc%FAIL, trim(message), ' at line ',line_num,bc%ENDC
    else
      write(6,*) bc%FAIL, trim(message), bc%ENDC    
      write(0,*) bc%FAIL, trim(message), bc%ENDC
    endif
  endif
      

#ifdef _WIN32
  read(5,'(a)',iostat=ios) ! allows to see the message !
#endif

#ifndef TURNOFF_STOP
  stop 'die'
#endif  

end subroutine ! die


end module !m_die

