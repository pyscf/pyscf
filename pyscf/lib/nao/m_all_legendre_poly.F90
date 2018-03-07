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

module m_all_legendre_poly

#include "m_define_macro.F90" 
  
  implicit none

  contains

!
! Computes all Legendre polynomials up to a given degree on a given grid
!
subroutine all_legendre_poly(xgla,ord,xj2p,jmax)
  use m_die, only : die
  use iso_c_binding, only: c_double, c_int
  implicit none
  !! external
  integer(c_int), intent(in) :: ord, jmax
  real(c_double), intent(in) :: xgla(ord) ! given grid
  real(c_double), intent(inout) :: xj2p(ord,jmax+1) ! Legengre polynomials
  !! internal
  integer :: k ! k = kappa+1

  if(ord<1) _die('ord<1')
  if(jmax<0) _die('jmax<0')
  
  xj2p(1:ord,1)=1.D0
  xj2p(1:ord,2)=xgla(1:ord)
  do k=2,jmax
    xj2p(:,k+1)=((2*k-1)*xgla*xj2p(:,k)-(k-1)*xj2p(:,k-1))/k
  end do

end subroutine ! all_legendre_poly


end module !m_all_legendre_poly
