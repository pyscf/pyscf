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

module m_numint

  implicit none

#include "m_define_macro.F90"
! but don's use _die or _warn or time measuremens, por favor

contains

!!-------------------------------------------------------------------
!! Computes the knots and weights for Gauss-Legendre quadrature 
!! for an infinite range (-infinity..infinity)
!! (a transformation y = inf * 2 *( 1/(1-x)-1/(1+x) ) is applied)
!!-------------------------------------------------------------------
subroutine comp_xx_ww_gauleg_minf_pinf(inf, n, xx, ww)
  implicit none
  !! external
  real(8), intent(in) :: inf
  integer, intent(in) :: n
  real(8), intent(inout) :: xx(:),ww(:) 
  !! internal
  integer :: i
  
  call gauleg(n,xx,ww)
  do i=1,n
    ww(i) = 2 * inf * ww(i) * ( 1D0/(1D0-xx(i))**2 + 1D0/(1D0+xx(i))**2)
    xx(i) = 2 * inf *         ( 1D0/(1D0-xx(i))    - 1D0/(1D0+xx(i))   )
  enddo ! i

end subroutine ! comp_xx_ww_gauleg_minf_pinf


!!-------------------------------------------------------------------
!! Computes the knots and weights for Gauss-Legendre quadrature 
!! for semi-infinite range (a..infinity)
!! (a transformation y = I*2/(1-x)-I+b is applied)
!!-------------------------------------------------------------------
subroutine comp_xx_ww_gauleg_a_inf(a, inf, n, xx, ww)
  implicit none
  !! external
  real(8), intent(in) :: a, inf
  integer, intent(in)  :: n
  real(8), intent(inout) :: xx(:),ww(:) 
  !! internal
  integer :: i
  real(8) :: dinf, da
  
  call gauleg(n,xx,ww)
  
  dinf = (inf - a) / 2D0 / ((1D0/(1D0-xx(n))) - (1D0/(1D0-xx(1))))
  da   = a - 2*dinf/(1D0-xx(1)) + dinf
  
  do i=1,n
    ww(i) = 2D0*dinf*ww(i)/(1D0-xx(i))**2
    xx(i) = 2D0*dinf/(1D0-xx(i))-dinf+da
  enddo ! i

end subroutine ! comp_xx_ww_gauleg_a_inf

!!-------------------------------------------------------------------
!! Computes the knots and weights for Gauss-Legendre quadrature 
!! for semi-infinite range (a..b) (a transformation y = (b-a)*x/2+(b+a)/2 is applied)
!!-------------------------------------------------------------------
subroutine comp_xx_ww_gauleg_a_b(a, b, n, xx, ww)
  implicit none
  !! external
  real(8), intent(in) :: a,b
  integer, intent(in) :: n
  real(8), intent(inout) :: xx(:),ww(:) 

  call gauleg(n,xx,ww)
  xx = (b-a)*0.5D0*xx+(b+a)*0.5D0
  ww = ww * (b-a)*0.5D0
  
end subroutine ! comp_xx_ww_gauleg

!!-------------------------------------------------------------------
!! Computes the knots and weights for Gauss-Legendre quadrature 
!! for semi-infinite range (0..a..infinity)  with splitting of the interval in two 
!! sub intervals (0..a) and (a,infinity)
!!-------------------------------------------------------------------
subroutine comp_xx_ww_gauleg_0_inf(a, n1,n2, xx, ww)
  implicit none
  !! external
  real(8), intent(in) :: a
  integer, intent(in) :: n1,n2
  real(8), intent(inout) :: xx(:),ww(:) 
  !! internal

  call comp_xx_ww_gauleg_a_b(0D0, a, n1, xx(1:n1), ww(1:n1))
  call comp_xx_ww_gauleg_a_inf(a, 1D0, n2, xx(n1+1:n1+n2), ww(n1+1:n1+n2))
  
end subroutine ! comp_xx_ww_gauleg

!!-------------------------------------------------------------------
!! Calculate the knotes for a Gauss-Legendre procedure
!!-------------------------------------------------------------------
subroutine GL_knts(res, x1, x2, GLord, GLpcs)
  implicit none
  !! external
  real(8), intent(in) :: x1, x2
  integer, intent(in) :: GLord, GLpcs
  real(8), intent(inout) :: res(:)
  !! internal
  integer :: ipiece, iknote, ind;
  real(8) :: dx, bma2, bpa2, x2_p, x1_p;
  real(8), allocatable :: x(:), w(:)
  allocate(x(GLord))
  allocate(w(GLord))
  call gauleg(GLord,x,w)

  if (GLpcs<1) then; write(0,*)'err: GL_knts: GLpcs<1'; stop 1; endif;

  dx = (x2 - x1)/GLpcs;
  bma2 = dx*0.5D0;
  ind = 0;
  do ipiece=0, GLpcs-1
    x1_p = x1 + dx * ipiece;
    x2_p = x1 + dx * (ipiece+1);
    bpa2 = (x2_p+x1_p)*0.5D0;
    do iknote=1,GLord
      ind = ind + 1;
      res(ind) = bma2*x(iknote) + bpa2
    end do
  end do
  
  _dealloc(x)
  _dealloc(w)
  
end subroutine !GL_knts

!!-------------------------------------------------------------------
!! Calculate the weights for a Gauss-Legendre procedure
!! well, basically, it stores the weights differently
!!-------------------------------------------------------------------
subroutine GL_wgts(res, x1, x2, GLord, GLpcs)
  implicit none
  !! external
  real(8), intent(in) :: x1, x2
  integer, intent(in) :: GLord, GLpcs
  real(8), intent(inout) :: res(:)
  !! internal
  integer :: ipiece, iknote, ind;
  real(8) :: factor;
  real(8), allocatable :: x(:), w(:)
  allocate(x(GLord))
  allocate(w(GLord))
  call gauleg(GLord,x,w)

  if (GLpcs<1) then; write(0,*)'err: GL_wgts: GLpcs<1'; stop 1; end if;

  ind = 0;
  factor = (x2-x1)*0.5D0/GLpcs;
  do ipiece=0, GLpcs-1
    do iknote=1, GLord
      ind = ind + 1;
      res(ind) = w(iknote)*factor;
    end do
  end do

  _dealloc(x)
  _dealloc(w)

end subroutine !GL_wgts

!
! Generates the Gauss-Legendre knots and weights 
!
subroutine gauleg(n,x,w)
  implicit none
  !! external
  integer, intent(in) :: n
  real(8), intent(inout) :: x(:),w(:)
  !! internal
  real(8) :: eps,pi,z,p1,p2,p3,pp,z1
  integer :: i,m,j,c,cmx

  cmx = 10
  eps = 1.0D-15
  pi = 4.0D0*atan(1.0d0)

  m=(n+1)/2
  do i=1,m
    z=cos(pi*(i-0.25D0)/(n+0.5D0)) 

    do c=1,cmx
      p1=1.0D0
      p2=0.0D0
      do j=1,n
        p3=p2
        p2=p1
        p1=((2.0D0*j-1)*z*p2-(j-1)*p3)/j
      enddo ! j
      pp=n*(z*p1-p2)/(z*z-1.0D0)
      z1=z
      z=z1-p1/pp
      if (abs(z-z1)<eps) exit
    enddo ! c
    
    x(i)=-z
    x(n+1-i)=z
    w(i)=2.0D0/((1.0D0-z*z)*pp*pp)
    w(n+1-i)=w(i)
  enddo ! i

end subroutine !gauleg

!
! Generates the Gauss-Laguerre knots and weights for integration between 0 and infinity
!
subroutine gaulag(n,x,w)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: x(:),w(:)
  
  integer :: maxit,i,its,j
  real(8) :: eps,z,ai,p1,p2,p3,pp,z1
  
  maxit=20
  eps=3.0D-14
  z = 0
  do i=1,n
     if (i.eq.1) then
        z=3.0D0/(1.0D0+2.4D0*n)
     else if (i.eq.2) then
        z=z+15.0D0/(1.0D0+2.5D0*n)
     else
        ai=i-2
        z=z+((1.0D0+2.55D0*ai)/(1.9D0*ai))*(z-x(i-2))
     endif
     do its=1,maxit
        p1=1.0D0
        p2=0.0D0
        do j=1,n
           p3=p2
           p2=p1
           p1=((2*j-1-z)*p2-(j-1)*p3)/j
        end do
        pp=n*(p1-p2)/z
        z1=z
        z=z1-p1/pp
        if (abs(z-z1).le.eps) exit
     end do
     x(i)=z
     w(i)=-exp(z)/(pp*n*p2)
  end do

end subroutine !gaulag

!
!
!
subroutine alloc_init_knts_wgts_krack(n, knts, wgts)
  implicit none
  !! external
  integer, intent(in) :: n
  real(8), intent(inout), allocatable :: knts(:), wgts(:)
  !! internal
  
  _dealloc(knts)
  _dealloc(wgts)
  if(n<1) then
    write(0,*)__LINE__, __FILE__, 'n<1'
    stop 1
  endif  
    
  allocate(knts(n))
  allocate(wgts(n))
  
  call GL_knts(knts, -1D0, +1D0, n, 1)
  call GL_wgts(wgts, -1D0, +1D0, n, 1)
  wgts(1:n) = wgts(1:n) * ( 1D0 / log(2D0) / ( 1D0 - knts(1:n) ) )
  knts(1:n)  = log( 2D0 / (1D0 - knts(1:n) ) ) / log(2D0)

end subroutine ! alloc_init_knts_wgts_krack

end module !m_numint
