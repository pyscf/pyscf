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

module m_rsphar

  use iso_c_binding, only: c_double, c_int64_t
  
  implicit none

#include "m_define_macro.F90"
  
  integer(c_int64_t) :: lmx = -1
  real(c_double), allocatable  :: lm2aa(:), lm2bb(:), l2tlp1(:), l2tlm1(:), l21mhl(:)

  contains

!
!
!
subroutine init_rsphar(lmax)
  use m_fact, only : sgn, init_fact
  ! external
  integer(c_int64_t), intent(in) :: lmax
  !real(c_double), intent(inout), allocatable :: lm2aa(:), lm2bb(:), l2tlp1(:), l2tlm1(:), l21mhl(:)
  ! internal
  integer(c_int64_t) :: l,m,ind
  
  call dealloc_rsphar()
  lmx = lmax
  allocate(lm2aa(0:(lmx+1)**2-1))
  allocate(lm2bb(0:(lmx+1)**2-1))
  allocate(l2tlp1(0:lmx))
  allocate(l2tlm1(0:lmx))
  allocate(l21mhl(0:lmx))

  lm2aa = 0D0
  lm2bb = 0D0
  l2tlp1 = 0D0
  l2tlm1 = 0D0
  l21mhl = 0D0
  
  do l=0,lmx
    l2tlp1(l) = sqrt(2D0*l+1)
    if(l<1) cycle
    l21mhl(l) = sqrt((l-0.5D0)/l)
    l2tlm1(l) = sqrt(2D0*l-1)
  enddo
  
  do l=0,lmx
    do m=-l,l
      ind = l*(l+1)+m
      lm2aa(ind) = sqrt(1d0*l**2-m**2)
      lm2bb(ind) = sqrt(1d0*(l+1)**2-m**2)
    enddo
  enddo

  if(.not. allocated(sgn)) then
    call init_fact()
  endif
end subroutine !  

!
!
!
subroutine dealloc_rsphar()
  lmx = -1
  _dealloc(lm2aa)
  _dealloc(lm2bb)
  _dealloc(l2tlp1)
  _dealloc(l2tlm1)
  _dealloc(l21mhl)
end subroutine ! dealloc_rsphar

!
! real spherical harmonics fast and tested
! needs m_fact: call init_fact()
! warn: sgn in m_fact :-)
!
subroutine rsphar(r,lmax,res) bind(c, name='rsphar')
  use m_fact, only : onediv4pi, rttwo, sgn, pi

  implicit none 
  real(c_double), intent(in)  :: r(3)
  integer(c_int64_t), intent(in)  :: lmax
  real(c_double), intent(inout) :: res((lmax+1)**2)
  ! internal
  integer(c_int64_t) :: l,m,il1,il2,ind,ll2,twol,l2
  real(c_double) :: dd,phi,cc,ss,zz,cs,P,rt2lp1, xxpyy

  if(lmx<lmax) then
    ! probably better to raise an error here, and do the initialization
    ! before to call rsphar
    call init_rsphar(lmax)
    lmx = lmax
  endif
  
  xxpyy = r(1)*r(1)+r(2)*r(2);
  dd=sqrt(xxpyy+r(3)*r(3))

  if (dd < 1D-10) then
    res=0; 
    res(1)=onediv4pi
    return
  endif

  if (r(1) .eq. 0D0) then; 
    phi=0.5D0*pi; if (r(2).lt.0D0) phi=-phi;
  else; 
    phi=atan(r(2)/r(1)); if (r(1).lt. 0D0) phi=phi+pi 
  endif

  ss=sqrt(xxpyy)/dd 
  cc=r(3)/dd
  res(1)=onediv4pi;
  if (lmax.eq.0) return

  do l=1,lmax 
    twol = l+l;
    l2   = l*l;
    il2  = l2+twol;
    il1  = l2-1
    res(il2+1)=-ss*l21mhl(l)*res(il1+1) 
    res(il2)=cc*l2tlm1(l)*res(il1+1)
  end do

  if (lmax.ge.2) then
    do m=0,lmax-2
      if (m.lt.lmax) then
        do l=m+1,lmax-1
          ind=l*(l+1)+m 
          zz=(l+l+1)*cc*res(ind+1)-lm2aa(ind)*res(ind-l-l+1) 
          res(ind+l+l+2+1)=zz/lm2bb(ind)
        end do
      endif
    end do
  endif

  do l=0,lmax
    ll2=l*(l+1)
    res(ll2+1) = res(ll2+1)*l2tlp1(l)
    rt2lp1 = l2tlp1(l)*rttwo
    do m=1,l
      cs=sin(m*phi)
      cc=cos(m*phi)
      P = res(ll2+m+1)*sgn(m)*rt2lp1;
      res(ll2+m+1)=cc*P;
      res(ll2-m+1)=cs*P;
    enddo
  enddo
  return
end subroutine !rsphar


!
! real spherical harmonics 
!
subroutine rsphar_vec(r,nc,lmax,res) bind(c, name='rsphar_vec')
  use m_fact, only : sgn, init_fact
  implicit none
  integer(c_int64_t), intent(in)  :: nc
  real(c_double), intent(in)      :: r(3,nc)
  integer(c_int64_t), intent(in)  :: lmax
  real(c_double), intent(inout)   :: res((lmax+1)**2,nc) 
  ! internal
  integer(c_int64_t) :: ic
  
  if (nc<1) then
    write(6,*) __FILE__, __LINE__, nc
    stop 'nc<1'
  endif

  if(lmx<lmax) then
    call init_rsphar(lmax)
    lmx = lmax
  endif
  
  if(.not. allocated(sgn)) then
    call init_fact()
    !write(0,*) __FILE__, __LINE__
    !stop '!sgn'
  endif

  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP SHARED(nc, lmax, res, r) PRIVATE(ic)
  !$OMP DO
  do ic=1,nc
    call rsphar(r(:,ic), lmax, res(:,ic))
  enddo
  !$OMP END DO
  !$OMP END PARALLEL

end subroutine !rsphar_vec

!
!
!
subroutine rsphar_exp_vec(r,nc,lmax,res) bind(c, name='rsphar_exp_vec')
  implicit none
  integer(c_int64_t), intent(in)  :: nc
  real(c_double), intent(in)  :: r(nc,3)
  integer(c_int64_t), intent(in)  :: lmax
  real(c_double), intent(inout), target   :: res(nc,(lmax+1)**2) 
  ! internal
  real(c_double), allocatable :: norms(:)
  real(c_double), pointer :: x(:)=>null(), y(:)=>null(), z(:)=>null()
  real(c_double), pointer :: xy(:)=>null(), yz(:)=>null(), xz(:)=>null(), zz(:)=>null(), xxmyy(:)=>null()

  if ( nc<1 ) then
    write(6,*) __FILE__, __LINE__
    stop ' nc<1 '
  endif
  
  if ( lmax<0 ) then
    write(6,*) __FILE__, __LINE__
    stop ' lmax<0 '
  endif
  
!  write(6,*) __FILE__, __LINE__
!  write(6,*) ubound(r), lmax
!  write(6,*) ubound(res)
!  write(6,*) 
  
  ! l = 0 
  !$OMP WORKSHARE
  res(:,1) = 0.28209479177387D0
  !$OMP END WORKSHARE
  if (lmax<1) return

  allocate(norms(nc))

  norms = sqrt(r(:,1)*r(:,1)+r(:,2)*r(:,2)+r(:,3)*r(:,3))
  where (norms>0)
    res(:,2) = r(:,2) / norms
    res(:,3) = r(:,3) / norms
    res(:,4) = r(:,1) / norms
  else where 
    res(:,2) = 0 
    res(:,3) = 0
    res(:,4) = 0
  end where

  y => res(:,2)
  z => res(:,3)
  x => res(:,4)

  ! l = 1
  if (lmax<2) then
    res(:,2) = 0.48860251190292D0*y
    res(:,3) = 0.48860251190291D0*z
    res(:,4) = 0.48860251190292D0*x
    _dealloc(norms)
    return
  endif
    
  ! l = 2
  res(:,5) = x*y
  res(:,6) = y*z
  res(:,7) = z*z
  res(:,8) = x*z
  res(:,9) = (x**2-y**2)

  xy => res(:,5)
  yz => res(:,6)
  zz => res(:,7)
  xz => res(:,8)
  xxmyy => res(:,9)

  if (lmax<3) then
    res(:,2) = 0.48860251190292D0*y
    res(:,3) = 0.48860251190291D0*z
    res(:,4) = 0.48860251190292D0*x    
    res(:,5) = 1.092548430592079D0*xy
    res(:,6) = 1.092548430592079D0*yz
    res(:,7) = 0.31539156525252D0*(3*zz-1)
    res(:,8) = 1.092548430592079D0*xz
    res(:,9) = 0.54627421529603D0*(xxmyy)
    _dealloc(norms)
    return
  endif  
  
  ! l = 3
  res(:,10) = 0.59004358992664D0*(3*x**2*y-y**3)
  res(:,11) = 1.445305721320277D0*(2*x*y)*z
  res(:,12) = 0.30469719964297D0*y*(7.5D0*z**2-1.5D0)
  res(:,13) = 0.74635266518023D0*(2.5D0*z**3-1.5D0*z)
  res(:,14) = 0.30469719964297D0*x*(7.5D0*z**2-1.5D0)
  res(:,15) = 1.445305721320277D0*(-y**2+x**2)*z
  res(:,16) = 0.59004358992664D0*(-3*x*y**2+x**3)

  if (lmax<4) then
    res(:,2) = 0.48860251190292D0*y
    res(:,3) = 0.48860251190291D0*z
    res(:,4) = 0.48860251190292D0*x    
    res(:,5) = 1.092548430592079D0*xy
    res(:,6) = 1.092548430592079D0*yz
    res(:,7) = 0.31539156525252D0*(3*zz-1)
    res(:,8) = 1.092548430592079D0*xz
    res(:,9) = 0.54627421529603D0*(xxmyy)
    _dealloc(norms)
    return
  endif  

  ! l = 4
  res(:,17) = 0.62583573544917D0*(-4*x*y**3+4*x**3*y)
  res(:,18) = 1.77013076977993D0*(-y**3+3*x**2*y)*z
  res(:,19) = 0.063078313050504D0*(2*x*y)*(52.5D0*z**2-7.5D0)
  res(:,20) = 0.26761861742291D0*y*(17.5D0*z**3-7.5D0*z)
  res(:,21) = 0.84628437532163D0*(4.375D0*z**4-3.75D0*z**2+0.375D0)
  res(:,22) = 0.26761861742291D0*(x)*(17.5D0*z**3-7.5D0*z)
  res(:,23) = 0.063078313050504D0*(-y**2+x**2)*(52.5D0*z**2-7.5D0)
  res(:,24) = 1.77013076977993D0*(-3*x*y**2+x**3)*z
  res(:,25) = 0.62583573544917D0*(y**4-6*x**2*y**2+x**4)

  if (lmax<5) then
    res(:,2) = 0.48860251190292D0*y
    res(:,3) = 0.48860251190291D0*z
    res(:,4) = 0.48860251190292D0*x    
    res(:,5) = 1.092548430592079D0*xy
    res(:,6) = 1.092548430592079D0*yz
    res(:,7) = 0.31539156525252D0*(3*zz-1)
    res(:,8) = 1.092548430592079D0*xz
    res(:,9) = 0.54627421529603D0*(xxmyy)
    _dealloc(norms)
    return
  endif  
  
  stop '!impl'
 
!0.65638205684017*(1.0*y**5-1.2246467991473533E-15*x*y**4-10.0*x**2*y**3+1.2246467991473533E-15*x**3*y**2+5.0*x**4*y)
!2.075662314881041*(-2.4492935982947064E-16*y**4-4.0*x*y**3+7.347880794884119E-16*x**2*y**2+4.0*x**3*y)*z
!0.0093188247511476*(-1.0*y**3+3.6739403974420594E-16*x*y**2+3.0*x**2*y)*(472.5*z**2-52.49999999999998)
!0.04565273128546*(1.2246467991473532E-16*y**2+2.0*x*y)*(157.5*z**3-52.49999999999998*z)
!0.24157154730437*y*(39.375*z**4-26.24999999999999*z**2+1.875)
!0.93560257962738*(7.875*z**5-8.749999999999998*z**3+1.875*z)
!0.24157154730437*(6.123233995736766E-17*y+1.0*x)*(39.375*z**4-26.24999999999999*z**2+1.875)
!0.04565273128546*(-1.0*y**2+1.2246467991473532E-16*x*y+1.0*x**2)*(157.5*z**3-52.49999999999998*z)
!0.0093188247511476*(-1.8369701987210297E-16*y**3-3.0*x*y**2+1.8369701987210297E-16*x**2*y+1.0*x**3)*(472.5*z**2-52.49999999999998)
!2.075662314881041*(1.0*y**4-7.347880794884119E-16*x*y**3-6.0*x**2*y**2+2.4492935982947064E-16*x**3*y+1.0*x**4)*z
!0.65638205684017*(3.061616997868383E-16*y**5+5.0*x*y**4-1.8369701987210296E-15*x**2*y**3-10.0*x**3*y**2+3.061616997868383E-16*x**4*y+1.0*x**5)
!0.68318410519191*(3.6739403974420594E-16*y**6+6.0*x*y**5-3.67394039744206E-15*x**2*y**4-20.0*x**3*y**3+1.83697019872103E-15*x**4*y**2+6.0*x**5*y)
!2.366619162231752*(1.0*y**5-1.2246467991473533E-15*x*y**4-10.0*x**2*y**3+1.2246467991473533E-15*x**3*y**2+5.0*x**4*y)*z
!0.0010678622237644*(-2.4492935982947064E-16*y**4-4.0*x*y**3+7.347880794884119E-16*x**2*y**2+4.0*x**3*y)*(5197.5*z**2-472.5)
!0.0058489222826344*(-1.0*y**3+3.6739403974420594E-16*x*y**2+3.0*x**2*y)*(1732.5*z**3-472.5*z)
!0.035093533695806*(1.2246467991473532E-16*y**2+2.0*x*y)*(433.125*z**4-236.25*z**2+13.125)
!0.22195099524523*y*(86.625*z**5-78.75*z**3+13.125*z)
!1.017107236282054*(14.4375*z**6-19.6875*z**4+6.5625*z**2-0.3125)
!0.22195099524523*(6.123233995736766E-17*y+1.0*x)*(86.625*z**5-78.75*z**3+13.125*z)
!0.035093533695806*(-1.0*y**2+1.2246467991473532E-16*x*y+1.0*x**2)*(433.125*z**4-236.25*z**2+13.125)
!0.0058489222826344*(-1.8369701987210297E-16*y**3-3.0*x*y**2+1.8369701987210297E-16*x**2*y+1.0*x**3)*(1732.5*z**3-472.5*z)
!0.0010678622237644*(1.0*y**4-7.347880794884119E-16*x*y**3-6.0*x**2*y**2+2.4492935982947064E-16*x**3*y+1.0*x**4)*(5197.5*z**2-472.5)
!2.366619162231752*(3.061616997868383E-16*y**5+5.0*x*y**4-1.8369701987210296E-15*x**2*y**3-10.0*x**3*y**2+3.061616997868383E-16*x**4*y+1.0*x**5)*z
!0.68318410519191*(-1.0*y**6+1.83697019872103E-15*x*y**5+15.0*x**2*y**4-3.673940397442059E-15*x**3*y**3-15.0*x**4*y**2+3.6739403974420594E-16*x**5*y+1.0*x**6)
!0.70716273252459*(-1.0*y**7+2.5717582782094417E-15*x*y**6+21.0*x**2*y**5-8.572527594031471E-15*x**3*y**4-34.99999999999999*x**4*y**3+2.5717582782094417E-15*x**5*y**2+7.0*x**6*y)
!2.6459606618019*(3.6739403974420594E-16*y**6+6.0*x*y**5-3.67394039744206E-15*x**2*y**4-20.0*x**3*y**3+1.83697019872103E-15*x**4*y**2+6.0*x**5*y)*z
!9.983945718523528E-5*(1.0*y**5-1.2246467991473533E-15*x*y**4-10.0*x**2*y**3+1.2246467991473533E-15*x**3*y**2+5.0*x**4*y)*(67567.5*z**2-5197.500000000001)
!5.990367431114116E-4*(-2.4492935982947064E-16*y**4-4.0*x*y**3+7.347880794884119E-16*x**2*y**2+4.0*x**3*y)*(22522.5*z**3-5197.500000000001*z)
!0.0039735602250741*(-1.0*y**3+3.6739403974420594E-16*x*y**2+3.0*x**2*y)*(5630.625*z**4-2598.75*z**2+118.1249999999999)
!0.02809731380603*(1.2246467991473532E-16*y**2+2.0*x*y)*(1126.125*z**5-866.2500000000001*z**3+118.1249999999999*z)
!0.20647224590289*y*(187.6875*z**6-216.5625*z**4+59.06249999999998*z**2-2.187499999999999)
!1.092548430592079*(26.8125*z**7-43.31250000000001*z**5+19.68749999999999*z**3-2.187499999999999*z)
!0.20647224590289*(6.123233995736766E-17*y+1.0*x)*(187.6875*z**6-216.5625*z**4+59.06249999999998*z**2-2.187499999999999)
!0.02809731380603*(-1.0*y**2+1.2246467991473532E-16*x*y+1.0*x**2)*(1126.125*z**5-866.2500000000001*z**3+118.1249999999999*z)
!0.0039735602250741*(-1.8369701987210297E-16*y**3-3.0*x*y**2+1.8369701987210297E-16*x**2*y+1.0*x**3)*(5630.625*z**4-2598.75*z**2+118.1249999999999)
!5.990367431114116E-4*(1.0*y**4-7.347880794884119E-16*x*y**3-6.0*x**2*y**2+2.4492935982947064E-16*x**3*y+1.0*x**4)*(22522.5*z**3-5197.500000000001*z)
!9.983945718523528E-5*(3.061616997868383E-16*y**5+5.0*x*y**4-1.8369701987210296E-15*x**2*y**3-10.0*x**3*y**2+3.061616997868383E-16*x**4*y+1.0*x**5)*(67567.5*z**2-5197.500000000001)
!2.6459606618019*(-1.0*y**6+1.83697019872103E-15*x*y**5+15.0*x**2*y**4-3.673940397442059E-15*x**3*y**3-15.0*x**4*y**2+3.6739403974420594E-16*x**5*y+1.0*x**6)*z
!0.70716273252459*(-4.286263797015736E-16*y**7-7.0*x*y**6+6.429395695523605E-15*x**2*y**5+34.99999999999999*x**3*y**4-6.429395695523602E-15*x**4*y**3-21.0*x**5*y**2+4.286263797015736E-16*x**6*y+1.0*x**7)

end subroutine !

end module !m_rsphar
