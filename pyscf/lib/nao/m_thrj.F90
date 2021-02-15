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

module m_thrj

  implicit none

#include "m_define_macro.F90"

  contains

!
! Initializes auxiliary arrays (computes some irreducible part of 3-j coefficients)
!
subroutine init_thrj(lmax, ixxa, ixxb, ixxc, aa,na) bind(c, name='init_thrj')
  use m_fact, only : init_fact, fac, sgn
  use iso_c_binding, only: c_char, c_double, c_float, c_int64_t, c_int

  implicit none
  !! internal
  integer(c_int), intent(in)  :: lmax
  integer(c_int), intent(out) :: ixxa(lmax+1), ixxb(lmax+1), ixxc(lmax+1)
  integer(c_int), intent(in)  :: na
  real(c_double), intent(out) :: aa(na)
  
  !! external
  integer :: l1,l2,l3,m1,m2,m3,ii,ic,it,itmin,itmax,lg
  real(8) :: xx,ss,yyx
  integer :: no3j=-1

  if(.not. allocated(sgn)) then
    call init_fact()
    !write(0,*) __FILE__, __LINE__
    !stop '!sgn'
  endif

  !$OMP CRITICAL 
  call comp_number_of3j(lmax, no3j)
  if (no3j/=na) then
    write(6,*) __FILE__, __LINE__
    stop '! no3j/=na'
  endif
  do ii=0,lmax
    ixxa(ii+1)=ii*(ii+1)*(ii+2)*(2*ii+3)*(3*ii-1)/60
    ixxb(ii+1)=ii*(ii+1)*(3*ii**2+ii-1)/6
    ixxc(ii+1)=(ii+1)**2
  enddo
  
  ic=0
  yyx = 0
  do l1=0,lmax
    do l2=0,l1
      do m2=-l2,l2
        do l3=0,l2
          do m3=-l3,l3
            m1=-m2-m3
            if((l3.ge.l1-l2).and.(abs(m1).le.l1)) then
              ss=0.0d0
              lg=l1+l2+l3
              xx=fac(lg-2*l1)*fac(lg-2*l2)*fac(lg-2*l3)/fac(lg+1)
              xx=xx*fac(l3+m3)*fac(l3-m3)/(fac(l1+m1)*fac(l1-m1)*fac(l2+m2)*fac(l2-m2)) 
              itmin=max(0,l1-l2+m3) 
              itmax=min(l3-l2+l1,l3+m3) 
              do it=itmin,itmax
                ss=ss+sgn(it)*fac(l3+l1-m2-it)*fac(l2+m2+it) &
                     / (fac(l3+m3-it)*fac(it+l2-l1-m3)*fac(it)*fac(l3-l2+l1-it)) 
              enddo
              yyx=sgn(l2+m2)*sqrt(xx)*ss 
            endif
            ic=ic+1
            aa(ic)=yyx
          enddo
        enddo
      enddo
    enddo
  enddo
  !$OMP END CRITICAL

end subroutine !init_thrj


!
! Number of 3-j coefficients irreducible by symmetry 
!
subroutine comp_number_of3j(lmax, n3j)
  implicit none
  !! external
  integer, intent(in) :: lmax
  integer, intent(out) :: n3j
  
  !! internal
  integer :: l1,l2,l3,m2
  
  n3j=0
  do l1=0,lmax
    do l2=0,l1
      do m2=-l2,l2
        do l3=0,l2
          n3j=n3j+2*l3+1
        enddo
      enddo
    enddo
  enddo
  
end subroutine ! comp_number_of3j

end module !m_thrj
