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

module m_csphar

  implicit none

contains

!
! complex spherical harmonics fast and tested
! needs m_fact: call init_fact(iv)
! warn: sgn in m_fact :-)
!
subroutine csphar(r,res,lmax)
  use m_fact, only : sgn, onediv4pi, pi

  implicit none 
  !! external
  real(8), intent(in) :: r(:)
  integer, intent(in)  :: lmax
  complex(8), intent(inout) :: res(0:) ! output
  !! internal
  real(8) :: x,y,z,dd,phi,cc,ss,al,aa,bb,zz,cs,rt2lp1
  integer :: ll,l,m,il1,il2,ind,ll2

  x=r(1) 
  y=r(2) 
  z=r(3) 
  dd=sqrt(x*x+y*y+z*z)
  if (dd.lt.1.0d-10) then
     res(0)=onediv4pi
     ll=(lmax+1)**2-1 
     res(1:ll)=0.0D0 
     return
  endif

  if (x.eq.0) then
     phi=0.5D0*pi
     if (y.lt.0.0D0) phi=-phi 
  else
     phi=atan(y/x) 
     if (x.lt.0.0D0) phi=phi+pi 
  endif
  ss=sqrt(x*x+y*y)/dd 
  cc=z/dd
  res(0)=onediv4pi
  if (lmax.eq.0) return
  do l=1,lmax 
     al=l 
     il2=(l+1)**2-1 
     il1=l**2-1
     res(il2)=-ss*sqrt((al-0.5D0)/al)*res(il1) 
     res(il2-1)=cc*sqrt(2.0D0*al-1.0D0)*res(il1)
  end do
  if (lmax.ge.2) then
     do m=0,lmax-2
        if (m.lt.lmax) then
           do l=m+1,lmax-1
              ind=l*(l+1)+m 
              aa=l**2-m**2
              bb=(l+1)**2-m**2
              zz=(l+l+1.0D0)*cc*dble(res(ind))-sqrt(aa)*dble(res(ind-2*l)) 
              res(ind+2*(l+1))=zz/sqrt(bb) 
           end do
        endif
     enddo
  endif
  do l=0,lmax
     ll2=l*(l+1)
     rt2lp1=sqrt(l+l+1.0D0)
     do m=0,l
        cs=sin(m*phi)*rt2lp1
        cc=cos(m*phi)*rt2lp1
        res(ll2+m)=cmplx(cc,cs,8)*res(ll2+m)
        res(ll2-m)=sgn(m)*conjg(res(ll2+m))
     enddo
  enddo 
end subroutine !csphar


!
! complex spherical harmonics fast and tested
! needs m_fact: call init_fact(iv)
! warn: sgn in m_fact :-)
!  the way the Y_lm is different to csphar(...) subroutine:
!  Namely, in the csphar(...) subroutine the harmonics are stored
!    Y_lm(theta,phi) = ylm( l*(l+1)+m ),
!  while in the csphar_rev_m(...) subroutine the harmonics are stored
!    Y_lm(theta,phi) = ylm( l*(l+1)-m ),
!  where ylm(...) is the 1-d array computed by the suboutines.
! 
! This is used in the better vectorised codes as in the module m_pb_kernel_rows4
!
subroutine csphar_rev_m(r,ylm,lmax)
  implicit none 
  !! external
  real(8), intent(in) :: r(:)
  integer, intent(in)  :: lmax
  complex(8), intent(inout) :: ylm(0:)
  !! internal
  integer :: l,m,ll2
  complex(8) :: swp
  
  call csphar(r,ylm,lmax)
  
  do l=0,lmax
    ll2 = l*(l+1)
    do m=1,l
      swp = ylm(ll2+m)
      ylm(ll2+m) = ylm(ll2-m)
      ylm(ll2-m) = swp
    enddo 
  enddo

end subroutine !csphar_rev_m


end module !m_csphar
