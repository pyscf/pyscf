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

module m_sph_bes_trans

#include "m_define_macro.F90" 

  !! "sbt" stands for "spherical bessel transform"
  type Talman_plan_t
    integer    :: lmax = -999
    real(8), allocatable    :: premult(:), smallr(:), postdiv(:)
    real(8), allocatable    :: sbt_rr(:), sbt_kk(:), sbt_rr3(:), sbt_kk3(:)
    complex(8), allocatable :: mult_table1(:,:), mult_table2(:,:)
    real(8) :: sbt_rhomin = -999
    real(8) :: sbt_kappamin = -999
    real(8) :: sbt_rmin = -999
    real(8) :: sbt_kmin = -999
  end type !Talman_plan_t

  real(8), private    :: pi
  complex(8), private :: ci
  integer, private    :: nr=-999, nr2=-999

  !! threadprivate variables
  integer(8), private :: plan12, plan_r2c, plan_c2r
  complex(8), allocatable, private :: temp1(:), temp2(:), r2c_out(:), c2r_in(:)
  real(8), allocatable, private    :: r2c_in(:), c2r_out(:)
  !$OMP THREADPRIVATE(plan12,plan_r2c,plan_c2r,temp1)
  !$OMP THREADPRIVATE(temp2,r2c_in,r2c_out,c2r_in,c2r_out)

  contains

!
!
!
real(8) function sbt_rl_exp_rk(l,a,k)
  use m_fact, only : fac
  implicit none
  !! external
  integer, intent(in) :: l
  real(8), intent(in) :: a
  real(8), intent(in) :: k
  !! internal
  
  sbt_rl_exp_rk = 2D0**(l+1)*fac(l+1)*a*k**l / (k**2+a**2)**(l+2)
  
end function ! sbt_rl_exp_rk 


!
!
!
function get_jmx(p) result(jmx)
  implicit none
  !! external
  type(Talman_plan_t), target, intent(in) :: p
  integer :: jmx
  
  jmx = p%lmax

  if(jmx/=ubound(p%mult_table1,2)) then
    write(0,*) ' jmx/=ubound(mult_table1,2)', __FILE__, __LINE__
    stop 'jmx/=ubound(mult_table1,2)'
  endif

  if(jmx/=ubound(p%mult_table2,2)) then
    write(0,*) ' jmx/=ubound(mult_table2,2)', __FILE__, __LINE__
    stop 'jmx/=ubound(mult_table2,2)'
  endif

end function ! get_jmx  
  
  
!
! spherical bessel transform
!
subroutine sbt_execute(p,ff,gg,li,direction,np_in)
  implicit none
  !! External
  type(Talman_plan_t), target, intent(in) :: p
  real(8), intent(in) :: ff(:)
  real(8), intent(out) :: gg(:)
  integer, intent(in) :: li,direction
  integer, intent(in), optional :: np_in

  !! Internal
  integer :: i,kdiv, np
  real(8) :: factor,C,dr,rmin,kmin
  real(8), pointer :: ptr_rr3(:)

  if(li>p%lmax) then
    write(6,*) __FILE__, __LINE__, li, p%lmax
    stop 'sbt_execute: li>lmax'
  endif  
  if(li<0) stop 'sbt_execute: li<0'
  
  if(present(np_in)) then; np = np_in; else; np = 0; endif
  

  if (direction==1) then
    rmin     = p%sbt_rmin
    kmin     = p%sbt_kmin
    dr = log(p%sbt_rr(2)/p%sbt_rr(1))
    C = ff(1)/p%sbt_rr(1)**(np+li)
    ptr_rr3 => p%sbt_rr3
  else if (direction==-1) then
    rmin     = p%sbt_kmin
    kmin     = p%sbt_rmin
    dr = log(p%sbt_kk(2)/p%sbt_kk(1))
    C = ff(1)/p%sbt_kk(1)**(np+li)
    ptr_rr3 => p%sbt_kk3
  else
    write(6,*)"err: sbt_execute: direction=", direction
    stop
  endif

  ! Make the calculation for LARGE k values extend the input 
  ! to the doubled mesh, extrapolating the input as C r**(np+li)

  r2c_in(1:nr) = C*p%premult(1:nr)*p%smallr(1:nr)**(np+li)
  r2c_in(nr+1:nr2) = p%premult(nr+1:nr2)*ff(1:nr)
  call dfftw_execute(plan_r2c)

  ! obtain the large k results in the array gg
  temp1(1:nr) = conjg(r2c_out(1:nr))*p%mult_table1(1:nr,li)
  temp1(nr+1:nr2) = 0.0D0
  call dfftw_execute(plan12)
  factor = (rmin/kmin)**1.5D0
  gg(1:nr) = factor*real(temp2(nr+1:nr2))*p%postdiv(1:nr)


  ! obtain the SMALL k results in the array c2r_out
  r2c_in(1:nr) = ptr_rr3 * ff(1:nr)
  r2c_in(nr+1:nr2) = 0.0D0
  call dfftw_execute(plan_r2c)

  do i=1, nr+1; c2r_in(i) = conjg(r2c_out(i))* p%mult_table2(i,li); enddo;
  call dfftw_execute(plan_c2r)
  c2r_out(1:nr) = c2r_out(1:nr)*dr

  do i=1, nr; r2c_in(i)=abs(gg(i)-c2r_out(i)); enddo;
  kdiv = minloc(r2c_in(1:nr),1)
  gg(1:kdiv) = c2r_out(1:kdiv)

end subroutine !gsbt

!
!
!
subroutine sbt_plan(p, nr_in, lmax_in, rr_in, kk_in, with_sqrt_pi_2_in, fftw_flags_in)

  implicit none
#include <fftw3.f>

  !! external
  type(Talman_plan_t), intent(out) :: p
  integer, intent(in) :: nr_in, lmax_in
  real(8), intent(in) :: rr_in(:), kk_in(:)
  logical, intent(in), optional :: with_sqrt_pi_2_in
  integer, intent(in), optional :: fftw_flags_in
  !! internal
  integer :: i, it, ix, lk, ll, fftw_flags
  real(8), allocatable :: j_ltable(:,:), xj(:)
  real(8) :: factor,phi,phi1,phi2,phi3,rad,tt,aa,dt,xx,kappamin,&
    dr,rmin,kmin,rhomin
  logical :: with_sqrt_pi_2

  fftw_flags = FFTW_ESTIMATE
  if(present(fftw_flags_in)) fftw_flags = fftw_flags_in;

  with_sqrt_pi_2 = .true.
  if(present(with_sqrt_pi_2_in)) with_sqrt_pi_2 = with_sqrt_pi_2_in;

  if(nr_in<2) then; 
    write(6,*) 'sbt_plan nr_in<=1', nr_in
    stop
  endif

  if(lmax_in<0) then; 
    write(6,*) 'sbt_plan lmax_in<0',lmax_in 
    stop
  endif

  if(nr==-999) then
    nr = nr_in;
    nr2 = 2*nr_in
    !$OMP PARALLEL DEFAULT(NONE) SHARED(nr,nr2,p,lmax_in,fftw_flags)
    allocate(temp1(nr2))
    allocate(temp2(nr2))
    allocate(r2c_in(nr2))
    allocate(r2c_out(nr+1))
    allocate(c2r_in(nr+1))
    allocate(c2r_out(nr2))
    !$OMP CRITICAL
    call dfftw_plan_dft_1d(plan12,nr2,temp1,temp2,FFTW_BACKWARD,fftw_flags)
    call dfftw_plan_dft_r2c_1d(plan_r2c,nr2,r2c_in,r2c_out,fftw_flags);
    call dfftw_plan_dft_c2r_1d(plan_c2r,nr2,c2r_in,c2r_out,fftw_flags);
    !$OMP END CRITICAL
    !$OMP END PARALLEL
  else if (nr/=nr_in) then
    write(6,*) 'err: sbt_plan (nr/=nr_in)', nr, nr_in;
    write(6,*) 'cannot create new plan with nr/=', nr;
    write(6,*) 'this feature is not supported currently because';
    write(6,*) 'of a performance penalty in explicit threading';
    stop 
  endif

  if(lmax_in<0) then
    write(6,*) 'sbt_plan lmax<0', lmax_in
    stop;
  endif
  p%lmax = lmax_in

  allocate(p%sbt_rr(nr_in));
  allocate(p%sbt_kk(nr_in));
  allocate(p%sbt_rr3(nr_in));
  allocate(p%sbt_kk3(nr_in));
  allocate(p%premult(2*nr_in));
  allocate(p%smallr(nr_in));
  allocate(p%postdiv(nr_in));
  allocate(p%mult_table1(nr, 0:lmax_in));
  allocate(p%mult_table2(nr+1, 0:lmax_in));

  pi = 4.0D0*atan(1.0D0)
  ci = cmplx(0.0D0, 1.0D0, 8)

  p%sbt_rr(1:nr) = rr_in(1:nr)
  p%sbt_kk(1:nr) = kk_in(1:nr)
  p%sbt_rr3(1:nr) = rr_in(1:nr)**3
  p%sbt_kk3(1:nr) = kk_in(1:nr)**3

  rmin = rr_in(1)
  kmin = kk_in(1)  
  rhomin = log(rmin)
  kappamin = log(kmin)
  dr = log(rr_in(2)/rr_in(1))  
  dt = 2.0D0*pi/(nr2*dr) 

  p%sbt_rmin = rmin
  p%sbt_kmin = kmin  
  p%sbt_rhomin = rhomin
  p%sbt_kappamin = kappamin

  !! check FFTW
  temp1(:)=0.0D0
  temp1(1)=1.0D0
  call dfftw_execute(plan12)

  aa=nr2
  xx=sum(real(temp2))
  if (abs(aa-xx).gt.1d-10) then
     write (6,*) 'err: sbt_plan: problem with fftw sum(temp2):', sum(temp2)
     stop
  endif

  !   Obtain the r values for the extended mesh, and the values r_i^1.5
  !   in the arrays smallr and premult
  factor = exp(dr)
  p%smallr(nr) = rr_in(1)/factor 
  do i = nr-1,1,-1
     p%smallr(i) = p%smallr(i+1)/factor
  enddo
   
  factor = exp(1.5D0*dr) 
  p%premult(nr+1) = 1.0D0  
  do i = 2,nr
     p%premult(nr+i) = factor*p%premult(nr+i-1) 
  enddo
  p%premult(nr) = 1.0D0/factor
  do i = 2,nr  
     p%premult(nr-i+1) = p%premult(nr-i+2)/factor
  enddo

  !   Obtain the values 1/k_i^1/5 in the array postdivide
  p%postdiv(1) = 1.0D0
  if(with_sqrt_pi_2) p%postdiv(1) = 1.0D0/sqrt(pi/2)

  do i = 2,nr
     p%postdiv(i) = p%postdiv(i-1)/factor 
  enddo

  !   construct the array of M_l(t) times the phase
  do it = 1,nr
     tt = (it-1)*dt               ! Define a t value
     phi3 = (kappamin+rhomin)*tt  ! See Eq. (33)
     rad = sqrt(10.5D0**2+tt**2)
     phi = atan((2D0*tt)/21D0)
     phi1 = -10D0*phi-log(rad)*tt+tt+sin(phi)/(12D0*rad)&
          -sin(3D0*phi)/(360D0*rad**3)+sin(5D0*phi)/(1260D0*rad**5)&
          -sin(7D0*phi)/(1680D0*rad**7)
     do ix = 1,10
        phi = 2*tt/(2D0*ix-1)
        phi1 = phi1+atan((2D0*tt)/(2D0*ix-1))  ! see Eqs. (27) and (28)
     enddo

     if(tt>200d0) then
       phi2 = -atan(1d0)
     else  
       phi2 = -atan(sinh(pi*tt/2)/cosh(pi*tt/2))  ! see Eq. (20)
     endif  
     phi = phi1+phi2+phi3

     p%mult_table1(it,0) = sqrt(pi/2)*exp(ci*phi)/nr  ! Eq. (18)
     if (it.eq.1) p%mult_table1(it,0) = 0.5D0*p%mult_table1(it,0)
     phi = -phi2-atan(2D0*tt)
     if(p%lmax>0)p%mult_table1(it,1) = exp(2.0D0*ci*phi)*p%mult_table1(it,0) ! See Eq. (21)
      !    Apply Eq. (24)
     do lk = 1,p%lmax-1
        phi = -atan(2*tt/(2*lk+1))
        p%mult_table1(it,lk+1) = exp(2.0D0*ci*phi)*p%mult_table1(it,lk-1)
     enddo
  enddo

  !write(6,*)(p%mult_table1(1:3,0))
  !write(6,*)(p%mult_table1(nr-2:nr,0))

  !write(6,*)(p%mult_table1(1:3,1))
  !write(6,*)(p%mult_table1(nr-2:nr,1))

  !write(6,*)(p%mult_table1(1:3,2))
  !write(6,*)(p%mult_table1(nr-2:nr,2))

  !   make the initialization for the calculation at small k values
  !   for 2N mesh values

  allocate (j_ltable(nr2,0:p%lmax))
  allocate(xj(0:p%lmax))

  !   construct a table of j_l values

  do i = 1,nr2
     xx = exp(rhomin+kappamin+(i-1)*dr)  
     call XJL(xx,p%lmax,xj)
     do ll = 0,p%lmax
        j_ltable(i,ll) = xj(ll)
     enddo
  enddo

  do ll = 0,p%lmax
     temp1(1:nr2) = j_ltable(1:nr2,ll)
     call dfftw_execute(plan12)
     p%mult_table2(1:nr+1,ll) = conjg(temp2(1:nr+1))/nr2
  enddo
  if(with_sqrt_pi_2) p%mult_table2 = p%mult_table2/sqrt(pi/2)
  
  deallocate(j_ltable)
  deallocate(xj)
end subroutine !INITIALIZE

!
!
!
integer function get_nr(p)
  implicit none 
  type(Talman_plan_t), intent(in) :: p
  !
  integer :: nn(7)
  
  nn(1) = size(p%sbt_rr)
  nn(2) = size(p%sbt_kk)
  nn(3) = size(p%sbt_rr3)
  nn(4) = size(p%sbt_kk3)
  nn(5) = size(p%smallr)
  nn(6) = size(p%postdiv)
  nn(7) = size(p%mult_table1,1)

  get_nr = nn(1)
  if(any(nn<1)) then
    write(0,*) __FILE__, __LINE__
    stop '!nn<1'
  endif

  if(any(nn/=get_nr)) then
    write(0,*) __FILE__, __LINE__
    stop '!nn/=get_nr'
  endif
  
end function ! get_nr  
!
!
!
subroutine sbt_destroy(p)
  type(Talman_plan_t), intent(inout) :: p
  
  !$OMP PARALLEL
  !$OMP CRITICAL
  if(allocated(c2r_in)) then
    call dfftw_destroy_plan(plan12)
    call dfftw_destroy_plan(plan_r2c)
    call dfftw_destroy_plan(plan_c2r)
  endif
  !$OMP END CRITICAL

  _dealloc(temp1)
  _dealloc(temp2)
  _dealloc(r2c_in)
  _dealloc(r2c_out)
  _dealloc(c2r_in)
  _dealloc(c2r_out)
  !$OMP END PARALLEL
  nr = -999

  _dealloc(p%sbt_rr)
  _dealloc(p%sbt_kk)
  _dealloc(p%sbt_rr3)
  _dealloc(p%sbt_kk3)
  _dealloc(p%premult)
  _dealloc(p%smallr)
  _dealloc(p%postdiv)
  _dealloc(p%mult_table1)
  _dealloc(p%mult_table2)

end subroutine ! sbt_destroy

!
! Spherical bessel functions
! Computes a table of j_l(x) for fixed xx, Eq. (39)
!
subroutine XJL(xx,lc,xj) 
  implicit none
  integer :: k,l
  integer, intent(in) :: lc
  real(8), intent(in) :: xx
  real(8), intent(out) :: xj(0:lc)
  real(8) :: aam,aa,bbm,bb,sa,sb,qqm,aap,bbp,qq,cc
  real(8) :: sin_xx_div_xx

  xj = 0.0D0
  if (abs(xx) < 1.0d-10) then
    xj(0) = 1.0D0
    return
  endif

  sin_xx_div_xx = sin(xx)/xx;

  if (xx.lt.0.75D0*lc) then
     aam = 1.0D0
     aa = (2*lc+1)/xx
     bbm = 0.0D0
     bb = 1.0D0
     sa = -1.0D0
     qqm = 1.0D10
     do k = 1,50
        sb = (2*(lc+k)+1)/xx
        aap = sb*aa+sa*aam
        bbp = sb*bb+sa*bbm
        aam = aa
        bbm = bb
        aa = aap
        bb = bbp
        qq = aa/bb
        if (abs(qq-qqm).lt.1.0d-15) exit
        qqm = qq
     enddo
     xj(lc) = 1.0D0
     if (lc > 0) xj(lc-1) = qq
     if (lc > 1) then
        do l = lc-1,1,-1
           xj(l-1) = (2*l+1)*xj(l)/xx-xj(l+1)
        enddo
     endif
     cc = sin_xx_div_xx/xj(0)
     do l = 0,lc
        xj(l) = cc*xj(l)
     enddo
  else
     xj(0) = sin_xx_div_xx
     if (lc > 0) then
        xj(1) = xj(0)/xx-cos(xx)/xx
     endif
     if (lc > 1) then
        do l = 1,lc-1
           xj(l+1) = (2*l+1)*xj(l)/xx-xj(l-1)
        enddo
     endif
  endif
  return

end subroutine !XJL


!!
!! 6-point interpolation on the exponential mesh (J. Talman)
!!
subroutine sbt_interp(nr, f, x, y, rhomnx, dr)
  implicit none
  real(8), intent(in)  :: f(:), x, rhomnx, dr
  integer, intent(in)  :: nr
  real(8), intent(out) :: y

  ! internal variables
  real(8)  :: dy
  integer  :: k

  if(x<=0) then; y=f(1); return; endif

  k=int((log(x)-rhomnx)/dr+1)
  k=max(k,3)
  k=min(k,nr-3)

  dy=(log(x)-rhomnx-(k-1)*dr)/dr

  y=(-dy*(dy**2-1.0d0)*(dy-2.0d0)*(dy-3.0d0)*f(k-2)&
     +5.0d0*dy*(dy-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*f(k-1)&
     -10.0d0*(dy**2-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*f(k)&
     +10.0d0*dy*(dy+1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*f(k+1)&
     -5.0d0*dy*(dy**2-1.0d0)*(dy+2.0d0)*(dy-3.0d0)*f(k+2)&
     +dy*(dy**2-1.0d0)*(dy**2-4.0d0)*f(k+3))/120.0d0 
  return
end subroutine !sbt_interp

!
!
!
subroutine dealloc_sbt()
  implicit none 
  !$OMP PARALLEL
  _dealloc(temp1)
  _dealloc(r2c_out)
  _dealloc(c2r_in)
  _dealloc(c2r_out)
  _dealloc(r2c_in)
  _dealloc(temp2)
  !$OMP END PARALLEL
  nr = -999
  nr2 = -999

end subroutine ! dealloc_sbt


end module ! m_sph_bes_trans

