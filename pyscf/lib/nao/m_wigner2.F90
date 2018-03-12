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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!! Modul_Wigner2 !!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module m_wigner2

#include "m_define_macro.F90"
  use m_warn, only : warn

  public wigner
  public thrj
  public I3Y_real
  public I3Y_pk
  public crl_jt
  public cleb
  
  private
  
  real(8), allocatable, private :: aa(:) ! aa(1147185)
  integer, allocatable, private :: ixxa(:), ixxb(:), ixxc(:); ! ixxa(0:24),ixxb(0:24),ixxc(0:24)

  contains

!
!
!
subroutine comp_number_of3j(lmax, n3j)
  implicit none
  !! external
  integer, intent(in) :: lmax
  integer, intent(out) :: n3j
  
  !! internal
  integer :: l1,l2,l3,m2,m3
  
  n3j=0
  do l1=0,lmax
    do l2=0,l1
      do m2=-l2,l2
        do l3=0,l2
          do m3=-l3,l3
            n3j=n3j+1
          enddo
        enddo
      enddo
    enddo
  enddo
  
end subroutine ! comp_number_of3j

!
! 
!
function thrj(l1i,l2i,l3i,m1i,m2i,m3i)
  use m_fact, only : fac, sgn
  implicit none
  integer, intent(in) :: l1i,l2i,l3i,m1i,m2i,m3i
  real(8) :: thrj
  !! external
  integer :: l1,l2,l3,m1,m2,m3,ii,ic,iz,it,itmin,itmax,lg,icc
  real(8) :: xx,ss,ph,yyx
  integer, save :: istart
  data istart /0/
  integer, parameter :: lmax=34 ! how much 3-j coefficients will be stored
  integer :: no3j=-1

  thrj = 0d0
  if (abs(m1i)>l1i .or. abs(m2i)>l2i .or. abs(m3i)>l3i) return
  if (m1i+m2i+m3i /= 0) return

  if (istart.eq.0) then ! 
  !$OMP CRITICAL 
  if (istart.eq.0) then ! 
    istart=1
    allocate(ixxa(0:lmax), ixxb(0:lmax), ixxc(0:lmax));
    call comp_number_of3j(lmax, no3j);
    !write(,*) 'thrj (first call): no3j ', no3j
    allocate(aa(no3j));
    do ii=0,lmax
      ixxa(ii)=ii*(ii+1)*(ii+2)*(2*ii+3)*(3*ii-1)/60
      ixxb(ii)=ii*(ii+1)*(3*ii**2+ii-1)/6
      ixxc(ii)=(ii+1)**2
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
                  ss=ss+sgn(it)*fac(l3+l1-m2-it)*fac(l2+m2+it)&
                     /(fac(l3+m3-it)*fac(it+l2-l1-m3)*fac(it)*fac(l3-l2+l1-it)) 
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
  endif ! First initialization endif
  !$OMP END CRITICAL
  endif

  l1=l1i
  l2=l2i
  l3=l3i
  m1=m1i
  m2=m2i
  m3=m3i
  ph=1.0d0
  if (l1.lt.l2) then
     iz=l1
     l1=l2
     l2=iz
     iz=m1
     m1=m2
     m2=iz
     ph=ph*sgn(l1+l2+l3)
  endif
  if (l2.lt.l3) then
     iz=l2
     l2=l3
     l3=iz
     iz=m2
     m2=m3
     m3=iz
     ph=ph*sgn(l1+l2+l3)
  endif
  if (l1.lt.l2) then
     iz=l1
     l1=l2
     l2=iz
     iz=m1
     m1=m2
     m2=iz
     ph=ph*sgn(l1+l2+l3)
  endif
  if (l1.gt.lmax) then
     write (0,*) 'thrj: 3-j coefficient out of range ==>stop'
     stop
  endif

  if (l1>l2+l3) return

  icc=ixxa(l1)+ixxb(l2)+ixxc(l2)*(l2+m2)+ixxc(l3)-l3+m3   
  thrj=ph*aa(icc)
  return
end function !thrj

!
!
!
function wigner(j1,m1,j2,m2,j3,m3)
    ! Nur fuer ganzzahligen Input
    implicit none
    integer, intent(in) :: j1,m1,j2,m2,j3,m3
    real(8) :: wigner
    
    wigner=((-1D0)**(j1-j2-m3))*cleb(2*j1,2*m1,2*j2,2*m2,2*j3,-2*m3)/sqrt(dble(j3+j3+1))
end function wigner


function I3Y_standard(j1,m1,j2,m2,j3,m3)  ! komplexe Darstellung der Y
  implicit none
  Integer::j1,j2,j3,m1,m2,m3
  double precision::I3Y_standard,pi
  pi=4*atan(dble(1))
  I3Y_standard=sqrt(  (2*j1+1)*(2*j2+1)*(2*j3+1)/(4*pi) )*Wigner(j1,m1,j2,m2,j3,m3)*Wigner(j1,0,j2,0,j3,0)
  ! Messiah, C.16, p.1057. I3Y(j1,m1,j2,m2,J,M) ist als Integral ueber YYY definiert
end function I3Y_standard 

!
!
!
real(8) function I3Y_real(j1,m1,j2,m2,j3,m3)  ! reelle Darstellung  der Y 
  implicit none
  Integer :: j1,j2,j3,m1,m2,m3,n1,n2,n3
  complex(8) :: tmp
  tmp=0
  do n1=-j1,j1; do n2=-j2,j2; do n3=-j3,j3
    if  ( (abs(n1)==abs(m1)).and.(abs(n2)==abs(m2)).and.(abs(n3)==abs(m3))  ) then 
       tmp= tmp+convert(m1,n1)*convert(m2,n2)*convert(m3,n3)*I3Y_standard(j1,n1,j2,n2,j3,n3)
    endif
  enddo; enddo; enddo ! {n1,n2,n3}
  I3Y_real = real(tmp,8)
end function I3Y_real

!
!
!
function I3Y_pk(j1,m1,j2,m2,j3)  ! komplexe Darstellung der Y
  implicit none
  Integer :: j1,j2,j3,m1,m2
  real(8) :: I3Y_pk,pi
  pi=4*atan(dble(1))
  I3Y_pk=sqrt(  (2*j1+1)*(2*j2+1)*(2*j3+1)/(4*pi) )*(-1D0)**(m1+m2)*Wigner(j1,m1,j2,m2,j3,-m1-m2)*Wigner(j1,0,j2,0,j3,0)
  ! Messiah, C.16, p.1057. I3Y(j1,m1,j2,m2,J,M) ist als Integral ueber YYY definiert
end function !I3Y_pk

!
!
!
complex(8) function convert(m,n)
implicit none
integer::m,n
if (abs(m)/=abs(n))then; convert=0.d0; endif
if (abs(m)==abs(n)) then
   if (    m == 0     ) then; convert =1.d0; endif
   if ((m>0).and.(n>0)) then; convert =((-1D0)**m)*0.5d0*sqrt(2.d0);endif
   if ((m<0).and.(n<0)) then; convert =cmplx(0.d0,0.5d0*sqrt(2.d0),8);endif  
   if ((m>0).and.(n<0)) then; convert =0.5d0*sqrt(2.d0);endif
   if ((m<0).and.(n>0)) then; convert =((-1D0)**m)*cmplx(0.d0, -0.5d0*sqrt(2.d0),8);endif    
endif
end function !convert

!
!
!
function cleb(j1,m1,j2,m2,j,m)
  implicit none
  ! calculate a clebsch-gordan coefficient < j1/2 m1/2 j2/2 m2/2 | j/2 m/2 >
  ! arguments are integer and twice the true value. 
  integer, intent(in) :: j1,m1,j2,m2,j,m
  real(8) :: cleb
  ! internal
  real(8) :: factor,sums
  integer :: par,z,zmin,zmax

    ! some checks for validity (let's just return zero for bogus arguments)

    if (2*(j1/2)-int(2*(j1/2.0D0)) /= 2*(abs(m1)/2)-int(2*(abs(m1)/2.0D0)) .or. &
         2*(j2/2)-int(2*(j2/2.0D0)) /= 2*(abs(m2)/2)-int(2*(abs(m2)/2.0D0)) .or. &
         2*(j/2)-int(2*(j/2.0D0)) /= 2*(abs(m)/2)-int(2*(abs(m)/2.0D0)) .or. &
         j1<0 .or. j2<0 .or. j<0 .or. abs(m1)>j1 .or. abs(m2)>j2 .or.&
         abs(m)>j .or. j1+j2<j .or. abs(j1-j2)>j .or. m1+m2/=m) then
       cleb= 0.0D0
       !_warn('bogus argument!!')
    else
       factor = 0.0D0
       factor = binom(j1,(j1+j2-j)/2) / binom((j1+j2+j+2)/2,(j1+j2-j)/2)
       factor = factor * binom(j2,(j1+j2-j)/2) / binom(j1,(j1-m1)/2)
       factor = factor / binom(j2,(j2-m2)/2) / binom(j,(j-m)/2)
       factor = sqrt(factor)

       zmin = max(0,j2+(j1-m1)/2-(j1+j2+j)/2,j1+(j2+m2)/2-(j1+j2+j)/2)
       zmax = min((j1+j2-j)/2,(j1-m1)/2,(j2+m2)/2)

       sums=0.0D0
       do z = zmin,zmax
          par=1
          if(2*(z/2)-int(2*(z/2.0D0)) /= 0) par=-1
          sums=sums+par*binom((j1+j2-j)/2,z)*binom((j1-j2+j)/2,(j1-m1)/2-z)*&
               binom((-j1+j2+j)/2,(j2+m2)/2-z)
       end do

       cleb = factor*sums
    end if
end function !cleb

!
!
!
recursive function factorial(n) result(res)
    implicit none
    integer :: n
    double precision :: res
    if (n==0 .or. n==1) then
       res=1.d0
    else
       res=n*factorial(n-1)
    end if
end function factorial


recursive function binom(n,r) result(res)
    implicit none
    integer :: n,r
    double precision :: res
    if(n==r .or. r==0) then
       res = 1
    else if (r==1) then
       res = dble(n)
    else
       res = dble(n)/dble(n-r)*binom(n-1,r)
    end if
end function binom

!
! This is Gaunt coefficient ( must be very clever )
! used in mod_overlap, for instance
!
function crl_jt(l1i,l2i,l3i,m1i,m2i,m3i)
  use m_fact, only : sgn
  implicit none
  ! external
  real(8) ::  crl_jt
  integer, intent(in) :: l1i,l2i,l3i,m1i,m2i,m3i
  ! internal
  integer :: l1,l2,l3,m1,m2,m3,is,m1a,m2a,m3a
  
  crl_jt=0
  if (m1i*m2i*m3i.lt.0) return
  l1=l1i
  l2=l2i
  l3=l3i
  m1=m1i
  m2=m2i
  m3=m3i
  m1a=abs(m1)
  m2a=abs(m2)
  m3a=abs(m3)
  if (m1a.lt.m2a) then
     is=m1
     m1=m2
     m2=is
     is=l1
     l1=l2
     l2=is
     m1a=abs(m1)
     m2a=abs(m2)
  endif
  if (m2a.lt.m3a) then
     is=m2
     m2=m3
     m3=is
     is=l2
     l2=l3
     l3=is
     m2a=abs(m2)
     m3a=abs(m3)
  endif
  if (m1a.lt.m2a) then
     is=m1
     m1=m2
     m2=is
     is=l1
     l1=l2
     l2=is
     m1a=abs(m1)
     m2a=abs(m2)
  endif
  if (m1a.gt.l1) return
  if (m1a.ne.m2a+m3a) return
  if ((m3.eq.0).and.(m1.ne.m2)) return
  crl_jt=sgn(m1a)*thrj(l1,l2,l3,-m1a,m2a,m3a)*thrj(l1,l2,l3,0,0,0)
  if (m3a.gt.0) crl_jt=sqrt(0.5D0)*crl_jt
  if ((m2.lt.0).and.(m3.lt.0)) crl_jt=-crl_jt
  return
end function !crl_jt

end module !m_wigner2
