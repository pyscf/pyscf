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

module m_abramowitz

  interface gamma
    module procedure gamma_dcmplx
    module procedure gamma_dble
  end interface gamma

contains

real(8) function spherical_bessel(L,X)
  implicit none
  integer, intent(in)  :: L
  real(8), intent(in) :: X

  spherical_Bessel=real(BESSPH(L,X),8)

end function !Spherical_Bessel


!
!  RETURNS THE SPHERICAL BESSEL FUNCTION JL(X).
!  REF: ABRAMOWITZ AND STEGUN, FORMULAS 10.1.2 AND 10.1.19
!  WRITTEN BY J.SOLER (JSOLER AT EMDUAM11). NOV/89.
! 
complex(8) function BESSPH (L,X)
  IMPLICIT DOUBLE PRECISION (A-H,O-Z)
  DOUBLE PRECISION :: ZERO, ONE, TINY, X, X2, TERM, SUM, Y, FNP1, FNM1, FN
  INTEGER :: NTERMS, L, SWITCH, I, N

  PARAMETER (ZERO=0.D0,ONE=1.D0,TINY=1.D-15,NTERMS=100)
      SWITCH=MAX(1,2*L-1)
      IF (ABS(X).LT.SWITCH) THEN
!       USE POWER SERIES
         TERM=ONE
         DO 10 I=1,L
            TERM=TERM*X/(2*I+1)
   10    CONTINUE
         X2=X*X
         SUM=ZERO
         DO 20 I=1,NTERMS
            SUM=SUM+TERM
            TERM=(-TERM)*X2/(2*I*(2*I+2*L+1))
            IF (ABS(TERM).LT.TINY) GO TO 30
   20    CONTINUE
            !real(8)(6,*) 'BESSPH: SERIES HAS NOT CONVERGED. L,X=',L,X
            STOP 3
   30    BESSPH=SUM
      ELSE
!       USE EXPLICIT EXPRESSIONS OR RECURRENCE RELATION
         IF (L.EQ.0) THEN
            BESSPH=SIN(X)/X
         ELSEIF (L.EQ.1) THEN
            BESSPH=(SIN(X)/X-COS(X))/X
         ELSE
            Y=ONE/X
            FNM1=SIN(X)*Y
            FN=(FNM1-COS(X))*Y
            DO 40 N=1,L-1
               FNP1=(2*N+1)*Y*FN-FNM1
               FNM1=FN
               FN=FNP1
   40       CONTINUE
            BESSPH=FN
         ENDIF
      ENDIF
END FUNCTION BESSPH

!
!
!
complex(8) function gamma_dcmplx(z)
  implicit none
  complex(8), intent(in) :: z
  real(8) ::pi
  integer::i
  complex(8) :: w,w_inverse,g_value, log_gamma
  
  pi=4*atan(1D0)
  w=z+10.d0; w_inverse=1.d0/w
  ! asymptotic expansion
  log_gamma=(w-0.5d0)*log(w)-w + 0.5d0*log(2.d0*pi)+w_inverse/12.0d0-(w_inverse**3)/360.d0+ &
  &                                        (w_inverse**5)/1260.d0 -  (w_inverse**7)/1680.0d0
  g_value=exp(log_gamma)
  ! now find, by recursion, from gamma(10+z) the value of gamma(z)
  do i=0,9; g_value=g_value/(i+z); enddo
  gamma_dcmplx=g_value
end function !gamma_dcmplx

!
!
!
real(8) function gamma_dble(z)
  implicit none
  real(8), intent(in) :: z
  real(8) :: pi
  integer :: i
  real(8) :: w,w_inverse,g_value, log_gamma
  
  pi=4*atan(1D0)
  w=z+10.d0; w_inverse=1.d0/w
  ! asymptotic expansion
  log_gamma=(w-0.5d0)*log(w)-w + 0.5d0*log(2.d0*pi)+w_inverse/12.0d0-(w_inverse**3)/360.d0+ &
  &                                        (w_inverse**5)/1260.d0 -  (w_inverse**7)/1680.0d0
  g_value=exp(log_gamma)
  ! now find, by recursion, from gamma(10+z) the value of gamma(z)
  do i=0,9; g_value=g_value/(i+z); enddo
  gamma_dble=g_value
  !real(8)(6,*)'gamma_dble', gamma_dble
end function !gamma_dble

!
!
!
function erf(x) result(fn_val)
!-----------------------------------------------------------------------
!             evaluation of the real error function
! based upon a fortran 66 routine in the naval surface warfare center's
! mathematics library (1993 version).
! adapted by alan.miller @ vic.cmis.csiro.au
!-----------------------------------------------------------------------
implicit none

real(8), intent(in) :: x
real(8)             :: fn_val

! local variables

real(8), parameter :: c = .564189583547756D0, one = 1D0, half = 0.5D0, &
                        zero = 0.0D0
real(8), parameter ::  &
           a(5) = (/ .771058495001320d-04, -.133733772997339d-02, &
                     .323076579225834d-01,  .479137145607681d-01, &
                     .128379167095513d+00 /),  &
           b(3) = (/ .301048631703895d-02,  .538971687740286d-01,  &
                     .375795757275549d+00 /),  &
           p(8) = (/ -1.36864857382717d-07, 5.64195517478974d-01,  &
                      7.21175825088309d+00, 4.31622272220567d+01,  &
                      1.52989285046940d+02, 3.39320816734344d+02,  &
                      4.51918953711873d+02, 3.00459261020162d+02 /), &
           q(8) = (/  1.00000000000000d+00, 1.27827273196294d+01,  &
                      7.70001529352295d+01, 2.77585444743988d+02,  &
                      6.38980264465631d+02, 9.31354094850610d+02,  &
                      7.90950925327898d+02, 3.00459260956983d+02 /), &
           r(5) = (/  2.10144126479064d+00, 2.62370141675169d+01,  &
                      2.13688200555087d+01, 4.65807828718470d+00,  &
                      2.82094791773523d-01 /),  &
           s(4) = (/  9.41537750555460d+01, 1.87114811799590d+02,  &
                      9.90191814623914d+01, 1.80124575948747d+01 /)
real(8) :: ax, bot, t, top, x2
!-------------------------
ax = abs(x)

if (ax <= half) then
  t = x*x
  top = ((((a(1)*t + a(2))*t + a(3))*t + a(4))*t + a(5)) + one
  bot = ((b(1)*t + b(2))*t + b(3))*t + one
  fn_val = x*(top/bot)
  return
end if

if (ax <= 4.0D0) then
  top = ((((((p(1)*ax + p(2))*ax + p(3))*ax + p(4))*ax + p(5))*ax  &
        + p(6))*ax + p(7))*ax + p(8)
  bot = ((((((q(1)*ax + q(2))*ax + q(3))*ax + q(4))*ax + q(5))*ax  &
        + q(6))*ax + q(7))*ax + q(8)
  fn_val = half + (half - exp(-x*x)*top/bot)
  if (x < zero) fn_val = -fn_val
  return
end if

if (ax < 5.8D0) then
  x2 = x*x
  t = one / x2
  top = (((r(1)*t + r(2))*t + r(3))*t + r(4))*t + r(5)
  bot = (((s(1)*t + s(2))*t + s(3))*t + s(4))*t + one
  fn_val = (c - top/(x2*bot)) / ax
  fn_val = half + (half - exp(-x2)*fn_val)
  if (x < zero) fn_val = -fn_val
  return
end if

fn_val = sign(one, x)
return
end function !erf

! c       ===============================================
! c       purpose: compute legendre polynomials pn(x)
! c                and their derivatives pn'(x)
! c       input :  x --- argument of pn(x)
! c                n --- degree of pn(x) ( n = 0,1,...)
! c       output:  pn(n) --- pn(x)
! c                pd(n) --- pn'(x)
! c       ===============================================
! c
subroutine lpn(n,x,pn)
  implicit none 
  integer, intent(in)  :: n
  real(8), intent(in) :: x
  real(8), intent(out) :: pn(0:n)!,pd(0:n)
  !! internal variables
  integer  :: k
  real(8) :: p0, p1, pf
  
  pn(0)=1.0d0
  pn(1)=x
  p0=1.0d0
  p1=x
  do k=2,n
     pf=(2.0d0*k-1.0d0)/k*x*p1-(k-1.0d0)/k*p0
     pn(k)=pf
     p0=p1
     p1=pf
  enddo 
end subroutine !lpn

end module !m_abramowitz

