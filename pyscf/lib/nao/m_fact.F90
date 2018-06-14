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

module m_fact

#include "m_define_macro.F90"

  implicit none

  integer, parameter, private :: N = 170 ! set to 170 here because sunf90 showed FPE in factorial... (N=200)
  real(8), allocatable :: fac(:), sqrfac(:), sgn(:)
  real(8) :: rttwo, onediv4pi, pi
  
  contains

!
!
!
subroutine init_fact()
  implicit none 
  integer :: i

  if(allocated(fac)) then
    if(ubound(fac,1)/=N)deallocate(fac)
  endif  
  if(.not. allocated(fac) ) allocate( fac(0:N))
  if(allocated(sqrfac)) then
    if( ubound(sqrfac,1)/=N) deallocate(sqrfac)
  endif  
  if(.not. allocated(sqrfac) ) allocate( sqrfac(0:N))
  if(allocated(sgn)) then
    if(ubound(sgn,1)/=N)deallocate(sgn)
  endif  
  if(.not. allocated(sgn) ) allocate( sgn(-N:N))

  pi = 4D0*atan(1D0)
  
  fac(0)=1.0D0
  sgn(0)=1.0D0
  do i=1,N
    fac(i)=i*fac(i-1)
    sgn(i)=-sgn(i-1)
    sgn(-i)=sgn(i)
  enddo

  onediv4pi = 1.0D0/sqrt(4*pi);
  rttwo=sqrt(2.0D0)
  
end subroutine !init_fact

!
!
!
subroutine dealloc_fact()
  implicit none

  _dealloc(fac)
  _dealloc(sqrfac)
  _dealloc(sgn)  

end subroutine ! dealloc_fact

end module !m_fact
