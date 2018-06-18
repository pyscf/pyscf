! Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

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

