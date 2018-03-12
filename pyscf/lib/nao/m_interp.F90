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

module m_interp
  implicit none

#include "m_define_macro.F90"
  
  type interp_t
    integer :: nr = -999
    real(8) :: one_over_dr = -999
    real(8) :: dr = -999
    real(8) :: rho_min = -999
    real(8) :: rr1 = -999
    real(8) :: rr2 = -999
    real(8) :: rr1_sqr = -999
    integer :: grid_type = -999
  end type ! interp_t

  contains

!
!
!
subroutine init_interp(rr, a)
  implicit none
  !! external
  real(8), intent(in) :: rr(:)
  type(interp_t), intent(inout) :: a
  !! internal
  integer :: i, gtypes(2)
  real(8) :: dr, derr1, derr2, aerr2
  real(8), allocatable :: rr_model(:,:)  

  a%nr = size(rr)
  if(a%nr<1) then
    _where
    write(6,'(a,i17)')'%nr<1', a%nr
    stop 'init_interp'
  endif  

  gtypes = 0
  allocate(rr_model(a%nr,2))
  rr_model = -999
  
  dr = (rr(a%nr)-rr(1))/(a%nr-1)
  do i=1,a%nr; rr_model(i,2) = rr(1)+dr*(i-1); enddo
  derr2 = sum(abs(rr_model(:,2) - rr))/(a%nr)
  aerr2 = sum(abs(rr_model(:,2)*1d-16))
  if(derr2<aerr2) gtypes(2) = 1  

  if(gtypes(2)==0 .and. all(rr>0)) then ! try log grid
    dr = (log(rr(a%nr))-log(rr(1)))/(a%nr-1)
    do i=1,a%nr; rr_model(i,1) = rr(1)*exp(dr*(i-1)); enddo
    derr1 = 0
    do i=1,a%nr
      derr1 = derr1 + abs(rr_model(i,1) - rr(i))/(abs(rr_model(i,1) + rr(i)))
    enddo
    derr1 = derr1 / a%nr
    if(derr1<1d-12) gtypes(1) = 1
  endif  
  
  if(gtypes(1)==1) then ! i.e. log grid

    a%grid_type = 1
    a%dr = (log(rr(a%nr))-log(rr(1)))/(a%nr-1)
    a%rho_min = log(rr(1))
    a%one_over_dr  = 1D0 / a%dr
    a%rr1 = rr(1)
    a%rr1_sqr = rr(1)**2
    a%rr2 = rr(2)

  else if(gtypes(2)==1) then ! i.e. linear grid

    a%grid_type = 2
    a%dr = (rr(a%nr)-rr(1))/(a%nr-1)
    a%rho_min = rr(1)
    a%one_over_dr  = 1D0 / a%dr
    a%rr1 = rr(1)
    a%rr1_sqr = rr(1)**2
    a%rr2 = rr(2)

  else
    _where
    do i=1,min(a%nr,20) 
      write(6,'(i8,2x,3g20.12)') i, rr(i), rr_model(i,1), rr_model(i,2)
    enddo
    if(a%nr>20) then
      write(6,'(a)') '...'
      do i=a%nr-20,a%nr
       write(6,'(i8,2x,3g20.12)') i, rr(i), rr_model(i,1), rr_model(i,2)
      enddo
    endif
    
    write(6,*) derr1, derr2, aerr2
    stop 'unknown grid type'
  endif
    
  _dealloc(rr_model)
  
end subroutine ! init_interp



!
! used in gen_prod_val_fast_bilocal, gen_prod_val_fast_local
!
subroutine comp_coeff_m2p3_k(rhosquare, a, coeff, k)
  implicit none
  !! external
  real(8), intent(in)  :: rhosquare
  type(interp_t), intent(in) :: a
  real(8), intent(out) :: coeff(-2:3)
  integer, intent(out) :: k
  !! internal
  real(8) :: rho   
  real(8), parameter :: one120 = 1.0D0/120D0, one12=1.0D0/12D0, one24 = 1.0D0/24D0;

  ! interpolation
  real(8) :: logr, dy, dy2, dym3;

  select case(a%grid_type)
  
  case(1) ! i.e. log grid
  
    if(rhosquare<=a%rr1_sqr)then; 
      k = 3
      coeff = 0
      rho = sqrt(rhosquare)
      coeff(-2) = (rho - a%rr2) / (a%rr1 - a%rr2)
      coeff(-1) = (rho - a%rr1) / (a%rr2 - a%rr1)
      return    
    else;
      logr=log(rhosquare)*0.5D0;
    endif

    k=int((logr-a%rho_min)*a%one_over_dr+1)
    k=max(k,3)
    k=min(k,a%nr-3);
    dy= (logr-a%rho_min-(k-1)*a%dr)*a%one_over_dr;
    dy2  = dy*dy;
    dym3 = (dy-3.0d0);
    coeff(-2) = -dy*(dy2-1.0d0)*(dy-2.0d0) *dym3*one120;
    coeff(-1) =  dy*(dy-1.0d0) *(dy2-4.0d0)*dym3*one24;
    coeff(0)  =-(dy2-1.0d0)    *(dy2-4.0d0)*dym3*one12;
    coeff(1)  =  dy*(dy+1.0d0)*(dy2-4.0d0)*dym3*one12;
    coeff(2)  = -dy*(dy2-1.0d0)*(dy+2.0d0)*dym3*one24;
    coeff(3)  = dy*(dy2-1.0d0)*(dy2-4.0d0)*one120;
    
  case(2)  
    _where
    stop '!impl'
  case default
    _where
    stop 'unknown grid_type'
  end select  
        

end subroutine !comp_coeff2


!
! used in gen_prod_val_fast_bilocal, gen_prod_val_fast_local
!
subroutine comp_coeff_m2p3(a, rho, coeff, k)
  implicit none
  !! external
  type(interp_t), intent(in) :: a
  real(8), intent(in)  :: rho
  real(8), intent(out) :: coeff(-2:3)
  integer, intent(out) :: k
  !! internal
  real(8), parameter :: one120 = 1.0D0/120D0, one12=1.0D0/12D0, one24 = 1.0D0/24D0;

  ! interpolation
  real(8) :: logr, dy, dy2, dym3;

  select case(a%grid_type)
  case(1) ! i.e. log grid

    if(rho<=a%rr1)then; 
      k = 3
      coeff = 0
      coeff(-2) = (rho - a%rr2) / (a%rr1 - a%rr2)
      coeff(-1) = (rho - a%rr1) / (a%rr2 - a%rr1)
      return    
    else;
      logr=log(rho)
    endif

    k=int((logr-a%rho_min)*a%one_over_dr+1)
    k=max(k,3)
    k=min(k,a%nr-3);
    dy= (logr-a%rho_min-(k-1)*a%dr)*a%one_over_dr;
    dy2  = dy*dy;
    dym3 = (dy-3.0d0);
    coeff(-2) = -dy*(dy2-1.0d0)*(dy-2.0d0) *dym3*one120;
    coeff(-1) =  dy*(dy-1.0d0) *(dy2-4.0d0)*dym3*one24;
    coeff(0)  =-(dy2-1.0d0)    *(dy2-4.0d0)*dym3*one12;
    coeff(1)  =  dy*(dy+1.0d0)*(dy2-4.0d0)*dym3*one12;
    coeff(2)  = -dy*(dy2-1.0d0)*(dy+2.0d0)*dym3*one24;
    coeff(3)  = dy*(dy2-1.0d0)*(dy2-4.0d0)*one120;
    
  case(2) ! i.e. linear grid  
    k=nint((rho-a%rho_min)/a%dr+1D0)
    k=max(k,3)
    k=min(k,a%nr-3)
    dy=(rho-(k-1)*a%dr-a%rho_min)/a%dr
    dy2  = dy*dy
    dym3 = (dy-3.0d0)
    coeff(-2)=-dy*(dy2-1.0d0)*(dy-2.0d0)*(dym3)*one120
    coeff(-1)= dy*(dy-1.0d0)*(dy2-4.0d0)*(dym3)*one24
    coeff( 0)=-(dy2-1.0d0)*(dy2-4.0d0)*(dym3)*one12
    coeff( 1)=+dy*(dy+1.0d0)*(dy2-4.0d0)*(dym3)*one12
    coeff( 2)=-dy*(dy2-1.0d0)*(dy+2.0d0)*(dym3)*one24
    coeff( 3)=dy*(dy2-1.0d0)*(dy2-4.0d0)*one120

  case default
    _where
    stop 'unknown grid_type'
  end select  
  
end subroutine !comp_coeff_m2p3


end module !m_interp
