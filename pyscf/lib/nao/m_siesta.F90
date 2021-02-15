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

!/*
! * Author: Peter Koval <koval.peter@gmail.com>
! */

module m_siesta

  use iso_c_binding, only: c_char, c_double, c_int64_t

  implicit none

contains


!
!
!
subroutine c_get_string(fname_in) bind(c, name="c_get_string")
  implicit none
  !! external
  character(kind=c_char), intent(in) :: fname_in(*)
  !! internal
  integer :: c, i
  character(1000) :: fname
  fname = ""
  do i=1,len(fname)
    c = ichar(fname_in(i))
    if(c==0) exit
    fname(i:i) = char(c)
  enddo
  
  write(6,*) trim(fname), " ", __FILE__, __LINE__

end subroutine ! c_get_string


!
!
!
subroutine c_hello_world(a, d, n, dat) bind(c, name="c_hello_world")
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: a
  real(c_double), intent(in) :: d
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: dat(n)
  !! internal

  write(6,*) a,d,n,__FILE__, __LINE__
  dat(1) = 999

end subroutine ! c_hello_world

end module ! m_siesta
