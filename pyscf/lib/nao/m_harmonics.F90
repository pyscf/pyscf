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

module m_harmonics

#include "m_define_macro.F90"

  implicit none

contains

  ! Spherical_Bessel(L, X)
  ! legendre_pnm ( n, m, x)
  ! Y_lm(x,y,z)

!
!
!
function get_lu_slm(jmax) result(lu)
  implicit none
  !! external
  integer, intent(in) :: jmax
  integer :: lu(2)
  !! internal
  if(jmax<0) then
    write(6,*) __FILE__, __LINE__
    stop 'jmax<0'
  endif
  lu(1) = 0
  lu(2) = (jmax+1)**2-1
  
end function ! get_lu_slm


!
! Conversion from complex to real harmonics
!
subroutine init_c2r_hc_c2r(jcutoff, c2r, hc_c2r)
  implicit none
  ! external
  integer, intent(in) :: jcutoff
  complex(8), intent(inout), allocatable :: c2r(:,:), hc_c2r(:,:)
  ! internal
  integer :: m

  _dealloc(c2r)
  _dealloc(hc_c2r)  
  allocate(c2r(-jcutoff:jcutoff, -jcutoff:jcutoff))
  allocate(hc_c2r(-jcutoff:jcutoff, -jcutoff:jcutoff))
  c2r=0.d0
  c2r(0,0)=1.d0
  do m=1,jcutoff
     c2r( m, m)= ((-1D0)**m)* sqrt(0.5d0); 
     c2r( m,-m)= sqrt(0.5d0); 
     c2r(-m,-m)= cmplx(0.d0,  sqrt(0.5D0),8)
     c2r(-m, m)= ((-1D0)**m) *cmplx(0.d0, -sqrt(0.5d0),8)
  enddo
  hc_c2r = conjg(transpose(c2r))
  
end subroutine ! init_c2r_hc_c2r

!
!
!
subroutine comp_sph2cart(sph2cart)
  implicit none
  complex(8), intent(inout) :: sph2cart(1:3,-1:1)
  !! internal
  real(8) :: pi, c
  pi = 4D0*atan(1D0)

  !! Transformation matrix from spherical components to Cartesian coordinates
  sph2cart = 0;  c = sqrt(2*pi/3.0D0);
  sph2cart(1,-1) = c;
  sph2cart(1,1) = -c;
  sph2cart(2,-1) = cmplx(0D0, c,8); 
  sph2cart(2,1) = cmplx(0D0, c,8);
  sph2cart(3,0) = sqrt(4*pi/3D0);

  !call random_number(R2mR1);
  !allocate(ylm(0:(2*jmx+2)**2-1))
  !call csphar(R2mR1 , ylm, 2*jmx+1 );
  !dist  = sqrt(sum(R2mR1*R2mR1))
  !write(6,*) 'sum(abs(R2mR1-dist*matmul(sph2cart,ylm(1*(1+1)-1:1*(1+1)+1))))'
  !write(6,*) ylm(1*(1+1)-1:1*(1+1)+1)
  !write(6,*) sph2cart
  !write(6,*) sum(abs(R2mR1-dist*matmul(sph2cart,ylm(1*(1+1)-1:1*(1+1)+1))))
  !! END of Transformation matrix from spherical components to Cartesian coordinates
end subroutine ! comp_sph2cart


!
! Transformation of matrix elements calculated with complex spherical harmonics
! <Y_1|o|Y_2> to the matrix elements calculated with real spherical harmonics
!
subroutine c2r_mat(j1,j2,cmat,conjg_c2r,tr_c2r,dt,rmat,mat)
  implicit none
  !! external
  integer, intent(in) :: j1, j2, dt
  complex(8), intent(in) :: cmat(-j1:, -j2:)
  real(8), intent(out) :: rmat(-j1:, -j2:)
  complex(8), intent(in) :: conjg_c2r(-dt:, -dt:)
  complex(8), intent(in) :: tr_c2r(-dt:, -dt:)
  complex(8), intent(inout) :: mat(-j1:, -j2:)
  !! internal
  integer :: mm1, mm2

  do mm1=-j1,j1
    do mm2=-j2,j2
      if(mm2/=0) then
        mat(mm1, mm2) = cmat(mm1,mm2)*tr_c2r(mm2,mm2)+cmat(mm1,-mm2)*tr_c2r(-mm2,mm2)
      else
        mat(mm1, mm2) = cmat(mm1,mm2)*tr_c2r(mm2,mm2)
      endif
    enddo
  enddo

  do mm2=-j2,j2
    do mm1=-j1,j1
      if(mm1/=0) then
        rmat(mm1, mm2) = real(conjg_c2r(mm1,mm1)*mat(mm1,mm2)+conjg_c2r(mm1,-mm1)*mat(-mm1,mm2),8)
      else
        rmat(mm1, mm2) = real(conjg_c2r(mm1,mm1)*mat(mm1,mm2),8)
      end if
    enddo
  enddo

end subroutine ! c2r_mat


!
!
!
function legendre_pnm ( n, m, x)

  implicit none
!
!  Author: John Burkardt
!
  integer, intent(in) :: n,m
  real(8), intent(in) :: x
  real(8) :: legendre_pnm
  
  real(8) :: cx(0:n+1),fact,somx2
! bug im Originaltext: cx(0:n) Inkompatibel mit cx(m+1) = ... fuer m=n
  integer :: i
!
  if ( m < 0 ) then
    write ( *, * ) ' '
    write ( *, * ) 'LEGENDRE_PNM - Fatal error!'
    write ( *, * ) '  Input value of M is ', m
    write ( *, * ) '  but M must be nonnegative.'
    stop
  end if
 
  if ( m > n ) then
    write ( *, * ) ' '
    write ( *, * ) 'LEGENDRE_PNM - Fatal error!'
    write ( *, * ) '  Input values of M, N = ', m, n
    write ( *, * ) '  but M must be less than or equal to N.'
    stop
  end if
 
  if ( x < -1.0D+00 ) then
    write ( *, * ) ' '
    write ( *, * ) 'LEGENDRE_PNM - Fatal error!'
    write ( *, * ) '  Input value of X = ', x
    write ( *, * ) '  but X must be no less than -1.'
    stop
  end if
 
  if ( x > 1.0D+00 ) then
    write ( *, * ) ' '
    write ( *, * ) 'LEGENDRE_PNM - Fatal error!'
    write ( *, * ) '  Input value of X = ', x
    write ( *, * ) '  but X must be no more than 1.'
    stop
  end if
  
  cx(0:m-1) = dble(0.0D+00)

  cx(m) = dble(1.0D+00)
  somx2 = sqrt ( dble(1.0D+00) - x**2 )
 
  fact = dble(1.0D+00)
  do i = 1, m
    cx(m) = cx(m) * fact * somx2
    fact = fact + 2.0D+00
  end do
 
  cx(m+1) = x * dble ( 2 * m + 1 ) * cx(m)

  do i = m+2, n
    cx(i) = ( dble ( 2 * i - 1 ) * x * cx(i-1) &
      - dble ( i + m - 1 ) * cx(i-2) ) / dble ( i - m )
  end do
  legendre_pnm=cx(n)
  return
end function legendre_pnm




function Y_lm(l,m,vect)
  implicit none
  real(8), intent(in) :: vect(:)
  integer, intent(in) :: l,m
  complex(8) :: Y_lm
  ! internal
  integer :: m2, Kronecker
  real(8) :: x,y,z,r,cosinus,pi
  complex(8) :: Y2,phase
  real(8), parameter :: eps=1.0D-12
  logical :: singular, positive

  Y_lm = 0
  if (abs(m) > l) then; write(*,*)' Y_lm: abs(m) > l '; stop; end if
  x=vect(1);y=vect(2);z=vect(3);
  pi=4*atan(dble(1))
  r=sqrt(x*x+y*y+z*z);

  if(r==0 .and. l==0) then; 
    y_lm = 1.0D0/sqrt(4*pi); return;
  else if (r==0) then
    y_lm = 0.0D0; return;
  end if;


  cosinus=z/r;
  singular=((1- abs(cosinus)) < eps**2)

  if (singular) then
       Kronecker=0; if (m==0) then; Kronecker=1;endif
       positive=(cosinus >0)
           if (     positive) then; Y_lm=          Kronecker*sqrt(  (2*l+1)/(4*pi)  ); endif
           if (.not.positive) then; Y_lm=((-1D0)**l)*Kronecker*sqrt(  (2*l+1)/(4*pi)  ); endif
  endif ! singular

  if (.not.singular) then 
     phase=cmplx(x,y,8)/sqrt(x*x+y*y)
     m2=abs(m);
     Y2=((-1D0)**m)*sqrt(  ((2*l+1)/(4*pi)) * &
     fakultaet(l-m2)/dble(fakultaet(l+m2))  )*legendre_pnm(l,m2,cosinus)*(phase**m2)
     if (m.ge.0) then; Y_lm=Y2; end if  
     if (m.lt.0) then; Y_lm=((-1D0)**m)*conjg(Y2); end if
  endif ! .not.singular
  ! Durch Vergleich mit Mathematica bestaetigt - der Faktor =((-1)**m) ist notwendig 

End Function Y_lm


real(8) function Y_lm_real(j,m,vect)
  implicit none
  integer::j,m
  real(8)::vect(3),Y

  Y = 0
  if ( m > 0 )  then; Y=sqrt(2.d0)*real(Y_lm(j,     m,vect),8); endif     ! ohne exttra (-) **m  Faktot 
  if ( m < 0 )  then; Y=sqrt(2.d0)*aimag(Y_lm(j,abs(m),vect) ); endif
  if ( m==0  )  then; Y=real( Y_lm(j,m,vect),8 )                           ; endif
  ! Fuer reelle  Y_lm_real gibt es einen extra Faktor (-) **m  -> A.Blanco,M.Florez,M.Bermejo
  Y_lm_real=((-1D0)**m)*Y
end function Y_lm_real

!
! Set of complex spherical harmonics
!
subroutine csphar_df(r,res,lmax)

  implicit none
  integer, intent(in) :: lmax
  real(8), intent(in) :: r(3)
  complex(8), intent(out) :: res(0:)
  
  ! Internals
  integer :: l, m

  do l=0,lmax
  do m=-l,l
    res(l*(l+1)+m) = Y_lm(l,m,r)
  end do !m
  end do !j

end subroutine csphar_df

!
! Set of real spherical harmonics
!
subroutine rsphar_df(r,res,lmax)

  implicit none
  integer :: lmax
  double precision :: r(3)
  real(8) :: res(0:)
  ! Internals
  integer :: l, m

  do l=0,lmax
  do m=-l,l
    res(l*(l+1)+m) = Y_lm_real(l,m,r)
  end do !m
  end do !j

end subroutine rsphar_df

!
! real spherical harmonics fast and tested
! needs m_fact: call init_fact(iv)
! warn: sgn in m_fact :-)
!
subroutine rsphar(r,res,lmax)
  use m_fact, only : onediv4pi, rttwo, sgn, pi
  implicit none 
  real(8), intent(in)  :: r(3)
  integer, intent(in)   :: lmax
  real(8), intent(out) :: res(0:)
  ! internal
  integer :: l,m,il1,il2,ind,ll2,twol,l2
  real(8) :: dd,phi,cc,ss,aa,bb,zz,cs,P,rt2lp1, xxpyy

  xxpyy = r(1)*r(1)+r(2)*r(2);
  dd=sqrt(xxpyy+r(3)*r(3))

  if (dd < 1D-10) then
    res=0; 
    res(0)=onediv4pi
    return
  endif

  if (r(1) .eq. 0D0) then; 
    phi=0.5D0*pi; if (r(2).lt.0D0) phi=-phi;
  else; 
    phi=atan(r(2)/r(1)); if (r(1).lt. 0D0) phi=phi+pi 
  endif

  ss=sqrt(xxpyy)/dd 
  cc=r(3)/dd
  res(0)=onediv4pi;
  if (lmax.eq.0) return

  do l=1,lmax 
    twol = l+l;
    l2   = l*l;
    il2  = l2+twol;
    il1  = l2-1
    res(il2)=-ss*sqrt((l-0.5D0)/l)*res(il1) 
    res(il2-1)=cc*sqrt(twol-1D0)*res(il1)
  end do

  if (lmax.ge.2) then
    do m=0,lmax-2
      if (m.lt.lmax) then
        do l=m+1,lmax-1
          ind=l*(l+1)+m 
          aa=l**2-m**2
          bb=(l+1)**2-m**2
          zz=(l+l+1)*cc*res(ind)-sqrt(aa)*res(ind-l-l) 
          res(ind+l+l+2)=zz/sqrt(bb) 
        end do
      endif
    end do
  endif

  do l=0,lmax
    ll2=l*(l+1)
    rt2lp1=sqrt(l+l+1D0)
    res(ll2) = res(ll2)*rt2lp1
    do m=1,l
      cs=sin(m*phi); cc=cos(m*phi)
      P = res(ll2+m)*sgn(m)*rt2lp1*rttwo;
      res(ll2+m)=cc*P;
      res(ll2-m)=cs*P;
    enddo
  enddo
  return
end subroutine !rsphar

!
!
!
subroutine sphar_jt(r,res,lmax) 
  use m_fact, only : rttwo
  implicit none 
  !! external
  real(8), intent(in) :: r(3)
  real(8), intent(out) :: res(0:*)
  integer, intent(in) :: lmax
  
  !! internal
  real(8) :: x,y,z,dd,cc,ss,al,aa,bb,zz
  complex(8) :: c1,c2
  integer :: ll,l,m,il1,il2,ind,ll2
  
  x=r(1) 
  y=r(2) 
  z=r(3) 
  dd=sqrt(x*x+y*y+z*z)
  ll=(lmax+1)**2-1 
  do  l=1,ll 
     res(l)=0.0D0 
  end do
  if (dd.lt.1.0d-10) then
     res(0)=1.0d0
     return
  endif
  ss=sqrt(x*x+y*y)/dd 
  cc=z/dd
  res(0)=1.0d0
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
              zz=(2*l+1)*cc*res(ind)-sqrt(aa)*res(ind-2*l) 
              res(ind+2*(l+1))=zz/sqrt(bb) 
           end do
        endif
     end do
  endif
  dd=sqrt(x**2+y**2)
  if (dd.ne.0) then
     do l=1,lmax
        ll2=l*(l+1)
        c1=1.0d0
        c2=cmplx(x,y,8)/dd
        do m=1,l
           zz=res(ll2+m)
           c1=c2*c1
           res(ll2+m)=rttwo*real(c1,8)*zz
           res(ll2-m)=rttwo*aimag(c1)*zz
        end do
     end do
  endif
  return 
end subroutine !sphar_jt


function fakultaet(n)
    implicit none
    integer :: i, n
    real(8) :: fakultaet
    if (n < 0 ) then; write(*,*)' fakultaet: n<0 '
       STOP
    end if
   fakultaet=1
   do i=2,n
      fakultaet=i*fakultaet
   end do
end function !fakultaet


end module !m_harmonics
