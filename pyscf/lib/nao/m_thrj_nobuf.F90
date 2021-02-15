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

module m_thrj_nobuf

#include "m_define_macro.F90"
  use m_die, only : die
  use iso_c_binding, only: c_double, c_int64_t, c_int, c_double_complex

  implicit none
  private die

  contains

!
!
!
subroutine thrj_nobuf(l1,l2,l3,m1,m2,m3,thrjres) bind(c, name='thrj_nobuf')
  implicit none
  integer(c_int), intent(in) :: l1,l2,l3,m1,m2,m3   !! exernal
  real(c_double), intent(inout) :: thrjres
  thrjres = thrj(l1,l2,l3,m1,m2,m3)
end subroutine ! thrj_nobuf

!
! 3j, 6j, 9j symbols from Talman
!
function thrj(l1,l2,l3,m1,m2,m3)
  use m_fact, only : sgn, fac
  implicit none
  integer(c_int), intent(in) :: l1,l2,l3,m1,m2,m3   !! exernal
  real(c_double) :: thrj

  !! internal
  real(c_double) :: xx,cc,s,ph,t,dlt
  integer(c_int) :: lg,it,lgh,itmin,itmax
  thrj=0.0D0 
  if (m1**2.gt.l1**2) return 
  if (m2**2.gt.l2**2) return 
  if (m3**2.gt.l3**2) return 
  if (m1+m2+m3.ne.0) return 
  if (l3.lt.iabs(l1-l2)) return 
  if (l3.gt.l1+l2) return 
  lg=l1+l2+l3
  dlt=sqrt(fac(lg-2*l1)*fac(lg-2*l2)*fac(lg-2*l3)/fac(lg+1))
  if ((m1.eq.0).and.(m2.eq.0)) then
    if (mod(lg,2).eq.1) return
    lgh=lg/2
    thrj=sgn(lgh)*dlt*fac(lgh)/(fac(lgh-l1)*fac(lgh-l2)*fac(lgh-l3))
    return
  endif
  xx=fac(l3+m3)*fac(l3-m3)/(fac(l1+m1)*fac(l1-m1)&
       *fac(l2+m2)*fac(l2-m2)) 
  cc=dlt*sqrt(xx)
  itmin=max0(0,l1-l2+m3) 
  itmax=min0(l3-l2+l1,l3+m3) 
  s=0.0D0
  ph=1.0D0 
  it=itmin 
  if (mod(it+l2+m2,2).eq.1) ph=-ph 
  12 t=ph*fac(l3+l1-m2-it)*fac(l2+m2+it) 
  t=t/(fac(l3+m3-it)*fac(it+l2-l1-m3)*fac(it)*fac(l3-l2+l1-it)) 
  s=s+t 
  it=it+1 
  ph=-ph 
  if (it.le.itmax) go to 12 
  thrj=cc*s 
  return 
end function !thrj


end module !m_thrj_nobuf
