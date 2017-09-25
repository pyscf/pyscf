module m_timing
  implicit none

  !! for cputime_sys_clock
  integer(8), private :: rate=0
 
  contains

!!
!! call cputime(real(8) :: t)
!! defined in cputime.c
!!

!
! This function return a string of lenght 18 that contains date and time.
!
function get_cdatetime() result(cdatetime)
  implicit none
  !! external
  character(18)  :: cdatetime
  !! internal
  character( 8)  :: date
  character(10)  :: time
  character(10)  :: zone
  integer        :: values(8)
  
  call date_and_time(date, time, zone, values)
  
  cdatetime = date//time
  
end function !   get_cdatetime


!
! microseconds resolution? -- no
!
subroutine cputime_sys_clk(time)
  implicit none
  real(8), intent(out) :: time
  integer :: counter
  if(rate==0) call SYSTEM_CLOCK(COUNT_RATE=rate)
  call SYSTEM_CLOCK(COUNT=counter)
  time = counter*1.0D0/rate;
end subroutine !cputime


end module !m_timing

