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

module m_interpolation

  contains

!
! This is taken from dipc_collaboration project. Must be first derivative.
!
subroutine diff(ll,nr,za,ar,dr,zb)
  implicit none
  integer, intent(in)  :: nr, ll
  real(8), intent(in)  :: za(nr), ar(nr)
  real(8), intent(in)  :: dr
  real(8), intent(out) :: zb(nr)
  
  !! internal
  integer :: i
  
  zb(1)=ll*za(1)/ar(1)
  zb(2)=ll*za(2)/ar(2)
  zb(3)=ll*za(3)/ar(3)
  do i=4,nr-3
    zb(i)=(45.0d0*(za(i+1)-za(i-1))-9.0d0*(za(i+2)-za(i-2))&
      +za(i+3)-za(i-3))/(60.0d0*dr*ar(i))
  enddo
  zb(nr-2)=(za(nr)-za(nr-4)+8.0d0*(za(nr-1)-za(nr-3)))&
      / ( 12.0d0*dr*ar(nr-2) )
  zb(nr-1)=(za(nr)-za(nr-2))/(2.0d0*dr*ar(nr-1))
  zb(nr)=( 4.0d0*za(nr)-3.0d0*za(nr-1)+za(nr-2))/(2.0d0*dr*ar(nr) );
  return
end subroutine !diff

!
!
!
real(8) function get_dr_jt(grid_jt)
  !! external
  implicit none
  real(8), intent(in) :: grid_jt(:)
  !! internal
  integer :: nr
  nr = size(grid_jt)
  get_dr_jt = log(grid_jt(nr)/grid_jt(1))/(nr-1)
end function ! get_dr_jt 
  

!
! Computes parameters dr_jt, rho_min_jt which 
! only necessary to do interpolation on Talman's radial grid
!
subroutine grid_2_dr_and_rho_min(nr, grid_jt, dr_jt, rho_min_jt)
  !! external
  implicit none
  integer, intent(in) :: nr
  real(8), intent(in) :: grid_jt(*)
  real(8), intent(out) :: dr_jt, rho_min_jt !! Parameters of 6-point interpolation by Talman, for interp();

  !! internal
  
  dr_jt = (log(grid_jt(nr))-log(grid_jt(1)))/(nr-1)
  rho_min_jt = log(grid_jt(1))

end subroutine ! grid_2_dr_and_rho_min

!
! Driver subroutine to demonstrate interpolation subroutine 
!
subroutine interp_driver(grid_jt, ff, fname_plot_on_grid, fname_plot_interp, iv, ilog)
  use m_io, only : get_free_handle
  implicit none
  !! external
  real(8), intent(in) :: grid_jt(:)
  real(8), intent(in) :: ff(:,:)
  character(*), intent(in) :: fname_plot_on_grid, fname_plot_interp
  integer, intent(in) :: iv, ilog
  !! internal
  integer :: ifile, ir!, f
  real(8) :: x
  real(8), allocatable :: ff_f(:)
  real(8) :: dr_jt, rho_min_jt
  !! Dimensions
  integer :: nr, nf
  nr = size(grid_jt)
  nf = size(ff,2)
  !! END of Dimensions

  allocate(ff_f(nf))
  
  ifile = get_free_handle()
  open(ifile, file=fname_plot_on_grid, action='write');
  do ir=1,nr
    write(ifile,*) grid_jt(ir), ff(ir,1:nf)
  enddo
  close(ifile)
  if(iv>0)write(ilog,*) 'interp_driver: ', trim(fname_plot_on_grid)
  
  
  dr_jt = (log(grid_jt(nr))-log(grid_jt(1)))/(nr-1);
  rho_min_jt = log(grid_jt(1))

  ifile = get_free_handle()
  open(ifile, file=fname_plot_interp, action='write');

  do ir=1,nr
    x = ir*grid_jt(nr)/nr
    call interp_batch(nr,nf,ff,nr, x,dr_jt,rho_min_jt, ff_f);
    
    !do f=1,nf; call interp(ff(1:nr,f), x,ff_f(f), rho_min_jt,dr_jt, nr); enddo;
    write(ifile,*) x,ff_f(1:nf)
  enddo
  if(iv>0)write(ilog,*) 'interp_driver: ', trim(fname_plot_interp)

end subroutine ! interp_driver



!
! Bisectional  find
!
function find(nr, rr, r) result(ir)
  implicit none
  integer,  intent(in) :: nr
  real(8), intent(in) :: rr(nr)
  real(8), intent(in) :: r
  integer :: ir, i1, i2, ns, imid
  i1 = 1;
  i2 = nr;
  if(rr(1)>=r) then; ir=1; return; end if;

  ns = 0;
  do while(i2-i1>1)
    imid = (i2 + i1)/2;

    if((r - rr(imid))>0) then 
      i1 = imid;
    else
      i2 = imid;
    end if;
    ns = ns + 1
  end do
  ir = i2;

end function !find

!
!
!
real(8) function linear_interpolate(xx, yy, points, x)
  implicit none
  ! extern: 
  integer::points
  real(8)::xx(1:points),yy(1:points),x
  ! intern:
  integer::i
  real(8)::b
  real(8), parameter::small=1.0d-10
  logical::outside
  linear_interpolate = -999
  if (points < 2) then
     write(*,*) 'linear_interpolate: points < 2, stop'
     stop
  endif
  outside = ( (x >= xx(points)-small).or.(x <= xx(1)+small) )
  if (     outside) then; linear_interpolate = 0.d0; endif
  if (.not.outside) then
    i=points; b=xx(points);
    do while (b-x >0)
       i=i-1
       b=xx(i)
    enddo 
  linear_interpolate=yy(i)+(x-xx(i))*(yy(i+1) - yy(i))/(xx(i+1) - xx(i))
  endif
end function !linear_interpolate 

!
!
!
subroutine change_discretisation(xx_old,yy_old,n_old,xx_new,yy_new,n_new)
  implicit none
  integer::n_old, n_new
  real(8), dimension(1:n_old)::xx_old,yy_old
  real(8), dimension(1:n_new)::xx_new,yy_new
  ! intern:
  integer::i
  real(8)::x
  do i=1,n_new
   x=xx_new(i)
   yy_new(i)=lin(xx_old, yy_old, n_old,x)
  enddo 
end subroutine !change_discretisation

!
! More accurate than change_discretisation
!
subroutine change_discr_log(  yy_old, nr_old, &
  rho_min_old, delta_rho_old, xx_new, yy_new, nr_new)
  implicit none
  integer, intent(in)          :: nr_old, nr_new
  real(8), intent(in) :: rho_min_old, delta_rho_old
  real(8), dimension(1:nr_old), intent(in)  :: yy_old
  real(8), dimension(1:nr_new), intent(in)  :: xx_new
  real(8), dimension(1:nr_new), intent(out) :: yy_new
  ! intern:
  integer :: i
  real(8) :: x
  do i=1,nr_new
    x=xx_new(i)
    yy_new(i) = get_fval(yy_old, x, rho_min_old, delta_rho_old, nr_old);
  enddo
end subroutine !change_discr_log


function lin(rr, ff, nr, rho) result(res)
  implicit none
  integer,  intent(in)  :: nr
  real(8), intent(in)  :: rr(nr), ff(nr), rho
  real(8) :: res;
  integer  :: p, i;

  p = -1;
  do i=2, nr
    if(rho<rr(i) .and. rho>=rr(i-1)) then; p=i; exit; endif;
  end do;

  if (p>1) then;
    res = ff(p-1) + (rho-rr(p-1))*(ff(p)-ff(p-1))/(rr(p)-rr(p-1));
  else if (p==1) then;
    res = ff(p);
  else if (p==nr) then;
    res = ff(p);
  else
    res = 0;
  end if
end function !lin;


function lin_fast(rr, ff, nr, rho) result(res)
  implicit none
  integer,  intent(in)  :: nr
  real(8), intent(in)  :: rr(nr), ff(nr), rho
  real(8) :: res;

  integer :: p;

  p = find(nr, rr, rho);

  if (p>1) then;
    res = ff(p-1) + (rho-rr(p-1))*(ff(p)-ff(p-1))/(rr(p)-rr(p-1));
  else if (p==1) then;
    res = ff(p);
  else if (p==nr) then;
    res = ff(p);
  else
    res = 0;
  end if
end function !lin_fast;


!!
!! 6-point interpolation on the exponential mesh (J. Talman)
!!
real(8) function get_fval(ff, r, rho_min_jt, dr_jt, nr)
  implicit none
  !! external
  real(8), intent(in) :: ff(:), r, rho_min_jt, dr_jt
  integer, intent(in) :: nr

  !! internal
  real(8) :: dy
  integer  :: k

  if(r<=0) then; get_fval=ff(1); return; endif;

  k=int((log(r)-rho_min_jt)/dr_jt+1)
  k=max(k,3)
  k=min(k,nr-3)

  dy=(log(r)-rho_min_jt-(k-1)*dr_jt)/dr_jt

  get_fval =(-dy*(dy**2-1.0d0)*(dy-2.0d0)*(dy-3.0d0)*ff(k-2)&
       +5.0d0*dy*(dy-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*ff(k-1)&
       -10.0d0*(dy**2-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*ff(k)&
       +10.0d0*dy*(dy+1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*ff(k+1)&
       -5.0d0*dy*(dy**2-1.0d0)*(dy+2.0d0)*(dy-3.0d0)*ff(k+2)&
       +dy*(dy**2-1.0d0)*(dy**2-4.0d0)*ff(k+3))/120.0d0 

end function !interp

!!
!! 6-point interpolation on the exponential mesh (J. Talman)
!! Batch & BLAS-like version
!!
subroutine interp_batch(nr,nj,ix_j2fx,ldr, x,dr,rho_min, j2fx)
  implicit none
  !! external
  integer, intent(in) :: nr, nj, ldr
  real(8), intent(in) :: ix_j2fx(ldr,*), x,dr,rho_min
  real(8), intent(out) :: j2fx(*)

  !! internal
  real(8) :: dy
  integer  :: k

  if(x<=0) then; j2fx(1:nj)=ix_j2fx(1,1:nj); return; endif;

  k=int((log(x)-rho_min)/dr+1)
  k=max(k,3)
  k=min(k,nr-3)

  dy=(log(x)-rho_min-(k-1)*dr)/dr

  j2fx(1:nj)=(-dy*(dy**2-1.0d0)*(dy-2.0d0)*(dy-3.0d0)*ix_j2fx(k-2,1:nj) &
     + 5.0d0*dy*(dy-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*ix_j2fx(k-1,1:nj)   &
     - 10.0d0*(dy**2-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*ix_j2fx(k,1:nj)    &
     + 10.0d0*dy*(dy+1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*ix_j2fx(k+1,1:nj)  &
     - 5.0d0*dy*(dy**2-1.0d0)*(dy+2.0d0)*(dy-3.0d0)*ix_j2fx(k+2,1:nj)   &
     + dy*(dy**2-1.0d0)*(dy**2-4.0d0)*ix_j2fx(k+3,1:nj))/120.0d0; 

end subroutine !interp

!
! used in gen_prod_val_fast_bilocal, gen_prod_val_fast_local
!
subroutine comp_coeff2(coeff, k, rhosquare, nr, one_over_dr_jt, dr_jt, rho_min_jt)
  implicit none
  real(8), intent(out) :: coeff(-2:3)
  integer, intent(out) :: k
  real(8), intent(in)  :: rhosquare
  integer, intent(in)  :: nr
  real(8), intent(in)  :: one_over_dr_jt, dr_jt, rho_min_jt
  real(8), parameter :: one120 = 1.0D0/120D0, one12=1.0D0/12D0, one24 = 1.0D0/24D0;

  ! interpolation
  real(8) :: logr, dy, dy2, dym3;

  if(rhosquare<=0)then; 
    k = 3
    coeff(-2) = 1
    coeff(-1:3) = 0
    return
  else;
    logr=log(rhosquare)*0.5D0;
  endif

  k=int((logr-rho_min_jt)*one_over_dr_jt+1)
  k=max(k,3)
  k=min(k,nr-3);
  dy= (logr-rho_min_jt-(k-1)*dr_jt)*one_over_dr_jt;
  dy2  = dy*dy;
  dym3 = (dy-3.0d0);
  coeff(-2) = -dy*(dy2-1.0d0)*(dy-2.0d0) *dym3*one120;
  coeff(-1) =  dy*(dy-1.0d0) *(dy2-4.0d0)*dym3*one24;
  coeff(0)  =-(dy2-1.0d0)    *(dy2-4.0d0)*dym3*one12;
  coeff(1)  =  dy*(dy+1.0d0)*(dy2-4.0d0)*dym3*one12;
  coeff(2)  = -dy*(dy2-1.0d0)*(dy+2.0d0)*dym3*one24;
  coeff(3)  = dy*(dy2-1.0d0)*(dy2-4.0d0)*one120;

end subroutine !comp_coeff2


!
!
!
subroutine xint(nr,ar,dr,x,y,defi,iex)
  implicit none
  ! external
  integer, intent(in)  :: nr
  real(8), intent(in)  :: x(nr), ar(nr), dr
  real(8), intent(out) :: y(nr), defi
  ! internal
  integer iex,i
  real(8) :: tab(nr)
  
  y(1) = x(1)*ar(1)/(iex+1) 
  y(2) = x(2)*ar(2)/(iex+1)
  y(3) = x(3)*ar(3)/(iex+1)

  tab=x*ar*dr
  
  do i=4,nr-2
    y(i)=y(i-1)+(802.0d0*(tab(i)+tab(i-1))-93.0d0*(tab(i+1)+tab(i-2)) &
        +11.0d0*(tab(i+2)+tab(i-3)))/1440.0d0;
  enddo
  y(nr-1)=y(nr-2)+(13.0d0*(tab(nr-1)+tab(nr-2))-tab(nr)-tab(nr-3))/24.0d0
  y(nr)=y(nr-1)+0.5d0*(tab(nr)+tab(nr-1))
  defi = y(nr)
  
  return
  
end subroutine !xint


!!
!! 6-point interpolation on the exponential mesh (J. Talman)
!!
subroutine interp_sp(f,x,y,rhomnx,dr,nr)
  implicit none
  real(4), intent(in) :: f(*),x,rhomnx,dr
  real(4), intent(out) :: y
  integer, intent(in) :: nr

  ! internal variables
  real(8) :: dy
  integer  :: k

  if(x<=0) then; y=f(1); return; endif
  k=int((log(x)-rhomnx)/dr+1)
  k=max(k,3)
  k=min(k,nr-3)

  dy=(log(x)-rhomnx-(k-1)*dr)/dr

  y=real(-dy*(dy**2-1.0)*(dy-2.0)*(dy-3.0)*f(k-2)&
       +5.0*dy*(dy-1.0)*(dy**2-4.0)*(dy-3.0)*f(k-1)&
       -10.0*(dy**2-1.0)*(dy**2-4.0)*(dy-3.0)*f(k)&
       +10.0*dy*(dy+1.0)*(dy**2-4.0)*(dy-3.0)*f(k+1)&
       -5.0*dy*(dy**2-1.0)*(dy+2.0)*(dy-3.0)*f(k+2)&
       +dy*(dy**2-1.0)*(dy**2-4.0)*f(k+3))/120.0
  return
end subroutine !interp

!!
!! 6-point interpolation on an equidistant mesh
!!
subroutine interp_lin(f,rr,x,y,dr,nr)
  implicit none
  ! external
  real(8), intent(in) :: f(*), rr(*), x, dr
  integer, intent(in)  :: nr
  real(8), intent(out) :: y

  !! internal
  real(8) :: dy
  integer  :: k

  k=int((x+0.5D0*dr)/dr);
  if(k>nr)  then; y=0; return; endif;
  k=max(k,3)
  k=min(k,nr-3)
  !!$  write (6,*) k
  !!$  write (6,*) rr(k-1),x,rr(k)
  dy=(x-rr(k))/dr
  y=(-dy*(dy**2-1.0d0)*(dy-2.0d0)*(dy-3.0d0)*f(k-2)&
       +5.0d0*dy*(dy-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*f(k-1)&
       -10.0d0*(dy**2-1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*f(k)&
       +10.0d0*dy*(dy+1.0d0)*(dy**2-4.0d0)*(dy-3.0d0)*f(k+1)&
       -5.0d0*dy*(dy**2-1.0d0)*(dy+2.0d0)*(dy-3.0d0)*f(k+2)&
       +dy*(dy**2-1.0d0)*(dy**2-4.0d0)*f(k+3))/120.0d0 
  !!$  write (6,('(7f10.5)'))  rr(k-1),x,rr(k),f(k-1),y,f(k)
  !!$  write (6,('(7f10.5)')) rr(k-2),rr(k-1),rr(k),rr(k+1),rr(k+2)
  return
end subroutine !interp_lin


subroutine splint(delt,ya,y2a,n,x,y) 
!subroutine splint(delt,ya,y2a,n,x,y,dydx)
!! Cubic Spline Interpolation.
!! Adapted from Numerical Recipes for a uniform grid.

  implicit none
  !! external
  integer,  intent(in) :: n
  real(8), intent(in)  :: delt, ya(n), y2a(n), x
  real(8), intent(out) :: y
!  real(dp), intent(out) :: y, dydx

  !! internal
  integer  :: nlo, nhi
  real(8) :: a, b

  nlo=max(int(x/delt)+1,1)
  if(nlo>n-1) then; y=0; return; endif
  !if(nlo>n-1) then; y=0; dydx=0; return; endif
  nhi=min(nlo+1,n)
  a=nhi-x/delt-1
  b=1.0D0-a
  y=a*ya(nlo)+b*ya(nhi)+((a**3-a)*y2a(nlo)+(b**3-b)*y2a(nhi))*(delt**2)/6D0
  
!  dydx=(ya(nhi)-ya(nlo))/delt + (-((3*(a**2)-1._dp)*y2a(nlo))+ (3*(b**2)-1._dp)*y2a(nhi))*delt/6._dp
end subroutine ! splint


subroutine spline(delt,y,n,yp1,ypn,y2) 
!! Cubic Spline Interpolation.
!! Adapted from Numerical Recipes routines for a uniform grid
!! D. Sanchez-Portal, Oct. 1996.
!! Alberto Garcia, June 2000
!! Peter Koval, Dec 2009

  implicit none
  !! external
  integer, intent(in)    :: n
  real(8), intent(in)   :: delt, yp1, ypn, y(:)
  real(8), intent(out)  :: y2(:)

  !! internal
  integer i, k
  real(8) sig, p, qn, un

  real(8), allocatable  :: u(:)
  allocate(u(n));

  if (yp1.eq. huge(1D0)) then
    y2(1)=0
    u(1)=0
  else
    y2(1)=-0.5D0
    u(1)=(3.0D0/delt)*((y(2)-y(1))/delt-yp1)
  endif

  do i=2,n-1
    sig=0.5D0
    p=sig*y2(i-1)+2
    y2(i)=(sig-1)/p
    u(i)=(3*( y(i+1)+y(i-1)-2*y(i) )/(delt*delt)-sig*u(i-1))/p
  enddo

  if (ypn.eq.huge(1D0)) then
    qn=0
    un=0
  else
    qn=0.5D0
    un=(3/delt)*(ypn-(y(n)-y(n-1))/delt)
  endif

  y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1)
  do k=n-1,1,-1
    y2(k)=y2(k)*y2(k+1)+u(k)
  enddo

end subroutine !spline


end module !m_interpolation
