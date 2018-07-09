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

module m_prod_talman

#include "m_define_macro.F90"
  use m_die, only : die
  use iso_c_binding, only: c_double, c_int64_t, c_int, c_double_complex

  implicit none
  private die

!
! Calculation of expansion of products due to Talman
! [1] Talman JD. Multipole Expansions for Numerical Orbital Products
!  Int. J. Quant. Chem. 107, 1578--1584 (2007)
!
  contains

!
! Product reduction from Talman 
! phia -- Radial part of the first orbital (a)
! la   -- Angular momentum of orbital a
! Ra   -- Center of orbital a (cartesian vector)
! phib -- Radial part of the second orbital (b)
! lb   -- Angular momentum of orbital b
! Rb   -- Center of orbital b (cartesian vector)
! Rcen -- Center around which the product will be expanded
! lbdmxa -- Maximum value for capital Lambda
! rhotb  -- Table of radial functions (X_{j \Lambda \Lambda'} in [1])
! rr     -- Radial grid
! nr     -- Number of the grid's knotes
! jtb    -- Table of j quantum numbers
! clbdtb -- Table of capital Lambdas
! lbdtb  -- Table of Lambdas primes
! nterm  -- Number of entries in jtb, clbdtb, lbdtb
!
! Before calling to the prdred two things shall be done:
! 1. Initialize the auxilary tables (factorials, pi etc.) 
!    call init_fact(iv); !from the module m_fact
! 2. Initialize the jtb,clbdtb,lbdtb,nterm
!    This is done in two steps:
!     a. Compute nterm
!      nterm=0
!      ijmx=la+lb
!      do ij=abs(la-lb),ijmx
!         do clbd=0,lbdmxa
!            do lbd=abs(clbd-ij),clbd+ij
!               nterm=nterm+1
!            enddo
!         enddo
!      enddo
!      allocate(jtb(nterm),clbdtb(nterm),lbdtb(nterm),rhotb(nr,nterm))
!      write(6, *) 'nterm ', nterm
!
!     b. Fill the allocated tables
!      ix=0
!      do ij=abs(la-lb),ijmx
!        do clbd=0,jcutoff
!          do lbd=abs(clbd-ij),clbd+ij
!            ix=ix+1
!            jtb(ix)=ij
!            clbdtb(ix)=clbd
!            lbdtb(ix)=lbd
!          enddo
!        enddo
!      enddo
!

subroutine prdred_coeffs(phia,la,ra,phib,lb,rb,rcen,lbdmxa,rhotb,rr,nr,jtb,clbdtb,lbdtb,nterm,ord,pcs,rho_min_jt,dr_jt) &
  bind(c, name='prdred')
  use m_interp_coeffs, only : interp_coeffs
  use m_fact, only : fac, sgn
  use m_numint, only : gl_knts, gl_wgts
!  use m_timing, only : get_cdatetime
  implicit none
  !! external
  integer(c_int), intent(in)  :: nr, nterm, ord,pcs,la,lb
  real(c_double), intent(in)  :: phia(nr),phib(nr),rr(nr),ra(3),rb(3),rcen(3)
  integer(c_int), intent(in)  :: lbdtb(nterm),clbdtb(nterm),lbdmxa,jtb(nterm)
  real(c_double), intent(in)  :: rho_min_jt, dr_jt
  real(c_double), intent(out) :: rhotb(nr,nterm)
  
  !! internal
  real(8) :: ya(nr), yb(nr)
  real(8) :: yz(ord*pcs)
  real(8) :: raa,rbb,a1,a2,f1,f2,sumb,aa,bb,cc
  integer :: i,ix,ijmx,ij,clbd,kappa,kpmax,igla, lbd1_p_lbd2
  integer :: lbd1,lbdp1,lbd2,lbdp2,lc,lcmin,lcmax,lcp,lcpmin,lcpmax,clbdp

  real(8) :: plval(ord*pcs,0:2*lbdmxa+la+lb), fval(nr,0:2*lbdmxa+la+lb)
  real(8) :: xgla(ord*pcs), wgla(ord*pcs)
  real(8) :: ccs(6)!, rhomin_jt, dr_jt
  integer(c_int64_t) :: k

!  write(6,*) ' prdred    ', lbdmxa, 2*lbdmxa+la+lb
!  write(6,*) ' size(fac) ', size(fac)

  !rhomin_jt = log(rr(1))
  !dr_jt = log(rr(2)/rr(1))
  
  kpmax = -999
  ijmx=la+lb;
  call GL_knts(xgla, -1.0D0, 1.0D0, ord, pcs);
  call GL_wgts(wgla, -1.0D0, 1.0D0, ord, pcs);

  plval(:,0)=1.0D0
  plval(:,1)=xgla(:)
  do kappa=1,2*lbdmxa+ijmx-1
    plval(:,kappa+1)=((2*kappa+1)*xgla*plval(:,kappa)-kappa*plval(:,kappa-1))/(kappa+1)
  end do
  ya=phia/rr**la
  yb=phib/rr**lb

  raa=sqrt(sum((ra-rcen)**2))
  rbb=sqrt(sum((rb-rcen)**2))
  fval=0
  do i=1,nr
    do igla=1,pcs*ord
      a1=sqrt(rr(i)**2-2.0d0*raa*rr(i)*xgla(igla)+raa**2)
      call interp_coeffs(a1, int(nr,c_int64_t), rho_min_jt, dr_jt, k, ccs)
      f1 = sum(ya(k:k+5)*ccs)

      a2=sqrt(rr(i)**2+2.0d0*rbb*rr(i)*xgla(igla)+rbb**2)
      call interp_coeffs(a2, int(nr,c_int64_t), rho_min_jt, dr_jt, k, ccs)
      f2 = sum(yb(k:k+5)*ccs)

      yz(igla)=f1*f2
    enddo
    kpmax=0
    if (raa+rbb .gt. 1.0d-5) kpmax=2*lbdmxa+ijmx
    do kappa=0,kpmax
      fval(i,kappa)=0.5d0*sum(plval(:,kappa)*yz*wgla)
    enddo
  enddo


  rhotb=0.0D0
  do ix=1,nterm
     ij=jtb(ix)
     clbd=clbdtb(ix)
     clbdp=lbdtb(ix)
     do lbd1=0,la
        lbdp1=la-lbd1
        aa=thrj(lbd1,lbdp1,la,0,0,0)*fac(lbd1)*fac(lbdp1)*fac(2*la+1)&
             /(fac(2*lbd1)*fac(2*lbdp1)*fac(la))

        do lbd2=0,lb
           lbdp2=lb-lbd2
           bb=thrj(lbd2,lbdp2,lb,0,0,0)*fac(lbd2)*fac(lbdp2)*fac(2*lb+1)&
                /(fac(2*lbd2)*fac(2*lbdp2)*fac(lb))
           bb=aa*bb
           do kappa=0,kpmax
              sumb=0.0d0
              lcmin=max(abs(lbd1-lbd2),abs(clbd-kappa))
              lcmax=min(lbd1+lbd2,clbd+kappa)
              do lc=lcmin,lcmax,2
                 lcpmin=max(abs(lbdp1-lbdp2),abs(clbdp-kappa))
                 lcpmax=min(lbdp1+lbdp2,clbdp+kappa)
                 do lcp=lcpmin,lcpmax,2
                    if ((abs(lc-ij).le.lcp).and.(lcp.le.lc+ij)) then
                       sumb=sumb+(2*lc+1)*(2*lcp+1)&
                            *thrj(lbd1,lbd2,lc,0,0,0)&
                            *thrj(lbdp1,lbdp2,lcp,0,0,0)&
                            *thrj(lc,clbd,kappa,0,0,0)&
                            *thrj(lcp,clbdp,kappa,0,0,0)&
                            *sixj(clbd,clbdp,ij,lcp,lc,kappa)&
                            *ninej(la,lb,ij,lbd1,lbd2,lc,lbdp1,lbdp2,lcp)
                    endif
                 enddo
              enddo
              cc=sgn(lbd1+kappa+lb)*(2*ij+1)*(2*kappa+1)&
                   *(2*clbd+1)*(2*clbdp+1)*bb*sumb
              if (cc .ne. 0.0D0) then
                lbd1_p_lbd2 = lbd1 + lbd2
                rhotb(:,ix)=rhotb(:,ix)+cc*rr(1:nr)**(lbd1_p_lbd2)&
                      *dpowi(raa, lbdp1)*dpowi(rbb, lbdp2)*fval(:,kappa)
              endif
           enddo
        enddo
     enddo
  enddo
  return
end subroutine !prdred_coeffs

!
! Compute all interpolation coefficients needed for a given geometry (coordinates of atoms and center of expansion)
! the coefficient are also multiplied with square root of the integration weigths
!
subroutine all_interp_coeffs(ra,rb,rcen,rr,nr,xgla,sqrt_wgla,ord,ixrj2ck)
  use m_interp_coeffs, only : interp_coeffs
  implicit none
  !! external
  integer(c_int), intent(in) :: nr, ord
  real(c_double), intent(in) :: rr(nr),ra(3),rb(3),rcen(3),xgla(ord), sqrt_wgla(ord)
  real(c_double), intent(inout) :: ixrj2ck(7,ord,nr,2)
  !! internal
  integer :: ir, igla
  integer(c_int64_t) :: nr8, k
  real(c_double) :: a1, a2, raa, rbb, r2aa, r2bb, rho_min_jt, dr_jt

  nr8 = nr
  dr_jt = (log(rr(nr))-log(rr(1)))/(nr-1)
  rho_min_jt = log(rr(1))

  raa=sqrt(sum((ra-rcen)**2))
  rbb=sqrt(sum((rb-rcen)**2))
  do ir=1,nr
    r2aa = rr(ir)**2+raa**2
    r2bb = rr(ir)**2+rbb**2
    
    do igla=1,ord
      a1=(r2aa-2.0d0*raa*rr(ir)*xgla(igla))
      call interp_coeffs(sqrt(a1), nr8, rho_min_jt, dr_jt, k, ixrj2ck(1:6,igla,ir,1))
      ixrj2ck(1:6,igla,ir,1) = ixrj2ck(1:6,igla,ir,1)*sqrt_wgla(igla)
      ixrj2ck(7  ,igla,ir,1) = real(k, c_double)

      a2=(r2bb+2.0d0*rbb*rr(ir)*xgla(igla))
      call interp_coeffs(sqrt(a2), nr8, rho_min_jt, dr_jt, k, ixrj2ck(1:6,igla,ir,2))
      ixrj2ck(1:6,igla,ir,2) = ixrj2ck(1:6,igla,ir,2)*sqrt_wgla(igla) 
      ixrj2ck(7  ,igla,ir,2) = real(k, c_double)
    enddo
  enddo
  
end subroutine ! all_interp_coeffs

!
!
!
subroutine prdred_all_interp_coeffs(ya,la,ra,yb,lb,rb,rcen,lbdmxa,rhotb,rr,nr,jtb,clbdtb,lbdtb,nterm,ord,plval,jmax,ixrj2ck) &
  bind(c, name='prdred_all_interp_coeffs')
  use m_fact, only : fac, sgn
  implicit none
  !! external
  integer(c_int), intent(in)  :: nr, nterm,la,lb,ord,jmax
  real(c_double), intent(in)  :: ya(nr),yb(nr) ! radial orbitals / rr^l
  real(c_double), intent(in)  :: rr(nr),ra(3),rb(3),rcen(3)
  integer(c_int), intent(in)  :: lbdtb(nterm),clbdtb(nterm),lbdmxa,jtb(nterm)
  real(c_double), intent(in)  :: plval(ord,jmax+1)
  real(c_double), intent(in)  :: ixrj2ck(7,ord,nr,2)
  real(c_double), intent(inout) :: rhotb(nr,nterm)
  
  !! internal
  real(8) :: yz(ord)
  real(8) :: raa,rbb,f1,f2,sumb,aa,bb,cc,thrj1,thrj2
  real(8) :: t1,t2,tt(9)
  integer :: ir,ix,ij,clbd,kappa,kpmax,igla, lbd1_p_lbd2,k1,k2
  integer :: lbd1,lbdp1,lbd2,lbdp2,lc,lcmin,lcmax,lcp,lcpmin,lcpmax,clbdp
  real(8), allocatable :: fval(:,:)

!     write(6,*) 'prdred', lbdmxa, 2*lbdmxa+la+lb
  _t1
  tt = 0
  allocate(fval(nr,0:2*lbdmxa+la+lb))
  raa=sqrt(sum((ra-rcen)**2))
  rbb=sqrt(sum((rb-rcen)**2))
  kpmax=0
  if (raa+rbb .gt. 1.0d-5) kpmax=2*lbdmxa+la+lb

  fval=0
  do ir=1,nr
    do igla=1,ord
      k1 = int(ixrj2ck(7,igla,ir,1))
      !f1 = sum(ya(k1:k1+5)*ijxr2ck(1:6,1,igla,ir))
      !f1 = ddot(6, ya(k1),1, ijxr2ck(1,1,igla,ir),1)
      f1 = dot_product(ya(k1:k1+5), ixrj2ck(1:6,igla,ir,1))
      
      k2 = int(ixrj2ck(7,igla,ir,2))
      !f2 = sum(yb(k2:k2+5)*ijxr2ck(1:6,2,igla,ir))
      !f2 = ddot(6, yb(k2),1, ijxr2ck(1,2,igla,ir),1)
      f2 = dot_product(yb(k2:k2+5), ixrj2ck(1:6,igla,ir,2))
      
      yz(igla)=f1*f2
    enddo
    !yz = yz*wgla
    !call DGEMV('T', ord, kpmax+1,0.5d0,plval,ord,yz,1,0d0,fval(ir,0),nr)
    do kappa=0,kpmax; fval(ir,kappa)=0.5d0*dot_product(plval(:,kappa+1), yz); enddo
  enddo

  _t2(tt(1))

  rhotb=0.0D0
  do ix=1,nterm
     ij=jtb(ix)
     clbd=clbdtb(ix)
     clbdp=lbdtb(ix)
     do lbd1=0,la
        lbdp1=la-lbd1
        aa=thrj(lbd1,lbdp1,la,0,0,0)*fac(lbd1)*fac(lbdp1)*fac(2*la+1)&
             /(fac(2*lbd1)*fac(2*lbdp1)*fac(la))

        do lbd2=0,lb
           lbdp2=lb-lbd2
           bb=thrj(lbd2,lbdp2,lb,0,0,0)*fac(lbd2)*fac(lbdp2)*fac(2*lb+1)&
                /(fac(2*lbd2)*fac(2*lbdp2)*fac(lb))
           bb=aa*bb
           do kappa=0,kpmax
              sumb=0.0d0
              lcmin=max(abs(lbd1-lbd2),abs(clbd-kappa))
              lcmax=min(lbd1+lbd2,clbd+kappa)
              do lc=lcmin,lcmax,2
                 thrj1 = thrj(lbd1,lbd2,lc,0,0,0)
                 thrj2 = thrj(lc,clbd,kappa,0,0,0)

                 lcpmin=max(abs(lbdp1-lbdp2),abs(clbdp-kappa))
                 lcpmax=min(lbdp1+lbdp2,clbdp+kappa)
                 do lcp=lcpmin,lcpmax,2
                    if ((abs(lc-ij).le.lcp).and.(lcp.le.lc+ij)) then
                       sumb=sumb+(2*lc+1)*(2*lcp+1)&
                            * thrj1 &
                            * thrj(lbdp1,lbdp2,lcp,0,0,0) &
                            * thrj2 &
                            * thrj(lcp,clbdp,kappa,0,0,0) &
                            * sixj(clbd,clbdp,ij,lcp,lc,kappa) &
                            * ninej(la,lb,ij,lbd1,lbd2,lc,lbdp1,lbdp2,lcp)
                    endif
                 enddo
              enddo
              cc=sgn(lbd1+kappa+lb)*(2*ij+1)*(2*kappa+1)&
                   *(2*clbd+1)*(2*clbdp+1)*bb*sumb
              if (cc .ne. 0.0D0) then
                lbd1_p_lbd2 = lbd1 + lbd2
                rhotb(:,ix)=rhotb(:,ix)+cc*rr(1:nr)**(lbd1_p_lbd2) &
                      * dpowi(raa, lbdp1) * dpowi(rbb, lbdp2)*fval(:,kappa)
              endif
           enddo
        enddo
     enddo
  enddo
  
  _t2(tt(2))
  !write(6,*) __FILE__, __LINE__, tt(1:2)
 
  _dealloc(fval)
  
  return
end subroutine !prdred_all_interp_coeffs

!
! Interpolate all radial orbitals of a given specie according to a latter need in the subroutine prdred(...)
!
subroutine all_interp_values(m2ff,nr,nmu, ixrj2ck,ord, xrjm2v)
  implicit none
  !! external
  integer(c_int), intent(in) :: nr, ord, nmu
  real(c_double), intent(in) :: m2ff(nr,nmu)
  real(c_double), intent(in) :: ixrj2ck(7,ord,nr,2)
  real(c_double), intent(inout) :: xrjm2v(ord,nr,2,nmu)
  !! internal
  integer :: ir, igla, k1, k2, imu
  
  xrjm2v = 0D0
  do ir=1,nr
    do igla=1,ord
      k1 = int(ixrj2ck(7,igla,ir,1))
      k2 = int(ixrj2ck(7,igla,ir,2))
      
      do imu=1,nmu
        xrjm2v(igla,ir,1,imu) = dot_product(m2ff(k1:k1+5,imu), ixrj2ck(1:6,igla,ir,1))
        xrjm2v(igla,ir,2,imu) = dot_product(m2ff(k2:k2+5,imu), ixrj2ck(1:6,igla,ir,2))
      enddo ! imu
    enddo
  enddo

end subroutine ! all_interp_values

!
! Do product reduction with all needed values of radial orbitals precomputed
!
pure subroutine prdred_all_interp_values(xrj2f1,la,ra,xrj2f2,lb,rb,rcen,lbdmxa,rhotb,rr,nr,jtb,clbdtb,lbdtb,nterm,ord,plval,jmax,fval,yz) &
  bind(c, name='prdred_all_interp_values')
  use m_fact, only : fac, sgn
  implicit none
  !! external
  integer(c_int), intent(in)  :: nr, nterm,la,lb,ord,jmax
  real(c_double), intent(in)  :: xrj2f1(ord,nr,2) ! values of first radial orbital / rr^l as needed (ord,nr,2)
  real(c_double), intent(in)  :: xrj2f2(ord,nr,2) ! values of second radial orbital / rr^l as needed
  real(c_double), intent(in)  :: rr(nr),ra(3),rb(3),rcen(3)
  integer(c_int), intent(in)  :: lbdtb(nterm),clbdtb(nterm),lbdmxa,jtb(nterm)
  real(c_double), intent(in)  :: plval(ord,jmax+1)
  real(c_double), intent(inout) :: rhotb(nr,nterm)
  real(c_double), intent(inout) :: fval(nr,jmax+1),yz(ord)
  
  !! internal
  real(8) :: raa,rbb,sumb,aa,bb,cc,thrj1
!  real(8) :: t1,t2,tt(9)
  integer :: ir,ix,ij,clbd,k,kpmax,lbd1_p_lbd2,kappa
  integer :: lbd1,lbdp1,lbd2,lbdp2,lc,lcmin,lcmax,lcp,lcpmin,lcpmax,clbdp

! write(6,*) 'prdred', lbdmxa, 2*lbdmxa+la+lb
  !_t1
  !tt = 0
  raa=sqrt(sum((ra-rcen)**2))
  rbb=sqrt(sum((rb-rcen)**2))
  kpmax=0
  if (raa+rbb .gt. 1.0d-5) kpmax=2*lbdmxa+la+lb
  
  fval(1:nr,1:kpmax+1)=0
  do ir=1,nr
    yz(1:ord)=xrj2f1(:,ir,1)*xrj2f2(:,ir,2)
    do k=1,kpmax+1; fval(ir,k)=0.5d0*dot_product(plval(:,k), yz); enddo
  enddo
  
  !_t2(tt(1))

  rhotb=0.0D0
  do ix=1,nterm
     ij=jtb(ix)
     clbd=clbdtb(ix)
     clbdp=lbdtb(ix)
     do lbd1=0,la
        lbdp1=la-lbd1
        aa=thrj(lbd1,lbdp1,la,0,0,0)*fac(lbd1)*fac(lbdp1)*fac(2*la+1)&
             /(fac(2*lbd1)*fac(2*lbdp1)*fac(la))

        do lbd2=0,lb
           lbdp2=lb-lbd2
           bb=thrj(lbd2,lbdp2,lb,0,0,0)*fac(lbd2)*fac(lbdp2)*fac(2*lb+1)&
                /(fac(2*lbd2)*fac(2*lbdp2)*fac(lb))
           bb=aa*bb
           do kappa=0,kpmax
              sumb=0.0d0
              lcmin=max(abs(lbd1-lbd2),abs(clbd-kappa))
              lcmax=min(lbd1+lbd2,clbd+kappa)
              do lc=lcmin,lcmax,2
                 thrj1 = thrj(lbd1,lbd2,lc,0,0,0)*thrj(lc,clbd,kappa,0,0,0)
                 
                 lcpmin=max(abs(lbdp1-lbdp2),abs(clbdp-kappa))
                 lcpmax=min(lbdp1+lbdp2,clbdp+kappa)
                 do lcp=lcpmin,lcpmax,2
                    if ((abs(lc-ij).le.lcp).and.(lcp.le.lc+ij)) then
                       sumb=sumb+(2*lc+1)*(2*lcp+1)&
                            * thrj1 &
                            * thrj(lbdp1,lbdp2,lcp,0,0,0) &
                            * thrj(lcp,clbdp,kappa,0,0,0) &
                            * sixj(clbd,clbdp,ij,lcp,lc,kappa) &
                            * ninej(la,lb,ij,lbd1,lbd2,lc,lbdp1,lbdp2,lcp)
                    endif
                 enddo
              enddo
              cc=sgn(lbd1+kappa+lb)*(2*ij+1)*(2*kappa+1)&
                   *(2*clbd+1)*(2*clbdp+1)*bb*sumb
              if (cc .ne. 0.0D0) then
                lbd1_p_lbd2 = lbd1 + lbd2
                rhotb(:,ix)=rhotb(:,ix)+cc*rr(1:nr)**(lbd1_p_lbd2) &
                      * dpowi(raa, lbdp1) * dpowi(rbb, lbdp2)*fval(:,kappa+1)
              endif
           enddo
        enddo
     enddo
  enddo
  
  !_t2(tt(2))
  !write(6,'(a,i5,3x,2f9.4)') __FILE__, __LINE__, tt(1:2)
   
  return
end subroutine !prdred_all_interp_values


!
! Interpolate all radial orbitals of a given specie according to a latter need in the subroutine prdred(...)
!
subroutine all_interp_values1(m2ff,nr,nmu, ixr2ck,ord, xrm2v)
  implicit none
  !! external
  integer(c_int), intent(in) :: nr, ord, nmu
  real(c_double), intent(in) :: m2ff(nr,nmu)
  real(c_double), intent(in) :: ixr2ck(7,ord,nr)
  real(c_double), intent(inout) :: xrm2v(ord,nr,nmu)
  !! internal
  integer :: ir, igla, k, imu
  
  xrm2v = 0D0
  do ir=1,nr
    do igla=1,ord
      k = int(ixr2ck(7,igla,ir))
      
      do imu=1,nmu
        xrm2v(igla,ir,imu) = dot_product(m2ff(k:k+5,imu), ixr2ck(1:6,igla,ir))
      enddo ! imu
    enddo
  enddo

end subroutine ! all_interp_values1

!
! Do product reduction with all needed values of radial orbitals precomputed
!
pure subroutine prdred_all_interp_values1(xr2f1,la,ra,nrmx,xr2f2,lb,rb,rcen,lbdmxa,rhotb,rr,nr,jtb,clbdtb,lbdtb,nterm,ord,plval,jmax,fval,yz) &
  bind(c, name='prdred_all_interp_values1')
  use m_fact, only : fac, sgn
  implicit none
  !! external
  integer(c_int), intent(in)  :: nr, nterm,la,lb,ord,jmax,nrmx
  real(c_double), intent(in)  :: xr2f1(ord,nr) ! values of first radial orbital / rr^l as needed (ord,nr,2)
  real(c_double), intent(in)  :: xr2f2(ord,nr) ! values of second radial orbital / rr^l as needed
  real(c_double), intent(in)  :: rr(nr),ra(3),rb(3),rcen(3)
  integer(c_int), intent(in)  :: lbdtb(nterm),clbdtb(nterm),lbdmxa,jtb(nterm)
  real(c_double), intent(in)  :: plval(ord,jmax+1)
  real(c_double), intent(inout) :: rhotb(nr,nterm)
  real(c_double), intent(inout) :: fval(nr,jmax+1),yz(ord)
  
  !! internal
  real(8) :: raa,rbb,sumb,aa,bb,cc,thrj1
!  real(8) :: t1,t2,tt(9)
  integer :: ir,ix,ij,clbd,k,kpmax,lbd1_p_lbd2,kappa
  integer :: lbd1,lbdp1,lbd2,lbdp2,lc,lcmin,lcmax,lcp,lcpmin,lcpmax,clbdp

! write(6,*) 'prdred', lbdmxa, 2*lbdmxa+la+lb
  !_t1
  !tt = 0
  raa=sqrt(sum((ra-rcen)**2))
  rbb=sqrt(sum((rb-rcen)**2))
  kpmax=0
  if (raa+rbb .gt. 1.0d-5) kpmax=2*lbdmxa+la+lb
  
  fval(1:nr,1:kpmax+1)=0
  do ir=1,nrmx
    yz(1:ord)=xr2f1(:,ir)*xr2f2(:,ir)
    do k=1,kpmax+1; fval(ir,k)=0.5d0*dot_product(plval(:,k), yz); enddo
  enddo
  
  !_t2(tt(1))

  rhotb=0.0D0
  do ix=1,nterm
     ij=jtb(ix)
     clbd=clbdtb(ix)
     clbdp=lbdtb(ix)
     do lbd1=0,la
        lbdp1=la-lbd1
        aa=thrj(lbd1,lbdp1,la,0,0,0)*fac(lbd1)*fac(lbdp1)*fac(2*la+1)&
             /(fac(2*lbd1)*fac(2*lbdp1)*fac(la))

        do lbd2=0,lb
           lbdp2=lb-lbd2
           bb=thrj(lbd2,lbdp2,lb,0,0,0)*fac(lbd2)*fac(lbdp2)*fac(2*lb+1)&
                /(fac(2*lbd2)*fac(2*lbdp2)*fac(lb))
           bb=aa*bb
           do kappa=0,kpmax
              sumb=0.0d0
              lcmin=max(abs(lbd1-lbd2),abs(clbd-kappa))
              lcmax=min(lbd1+lbd2,clbd+kappa)
              do lc=lcmin,lcmax,2
                 thrj1 = thrj(lbd1,lbd2,lc,0,0,0)*thrj(lc,clbd,kappa,0,0,0)
                 
                 lcpmin=max(abs(lbdp1-lbdp2),abs(clbdp-kappa))
                 lcpmax=min(lbdp1+lbdp2,clbdp+kappa)
                 do lcp=lcpmin,lcpmax,2
                    if ((abs(lc-ij).le.lcp).and.(lcp.le.lc+ij)) then
                       sumb=sumb+(2*lc+1)*(2*lcp+1)&
                            * thrj1 &
                            * thrj(lbdp1,lbdp2,lcp,0,0,0) &
                            * thrj(lcp,clbdp,kappa,0,0,0) &
                            * sixj(clbd,clbdp,ij,lcp,lc,kappa) &
                            * ninej(la,lb,ij,lbd1,lbd2,lc,lbdp1,lbdp2,lcp)
                    endif
                 enddo
              enddo
              cc=sgn(lbd1+kappa+lb)*(2*ij+1)*(2*kappa+1)&
                   *(2*clbd+1)*(2*clbdp+1)*bb*sumb
              if (cc .ne. 0.0D0) then
                lbd1_p_lbd2 = lbd1 + lbd2
                rhotb(1:nrmx,ix)=rhotb(1:nrmx,ix)+cc*rr(1:nrmx)**(lbd1_p_lbd2) &
                      * dpowi(raa, lbdp1) * dpowi(rbb, lbdp2)*fval(1:nrmx,kappa+1)
              endif
           enddo
        enddo
     enddo
  enddo
  
  !_t2(tt(2))
  !write(6,'(a,i5,3x,2f9.4)') __FILE__, __LINE__, tt(1:2)
   
  return
end subroutine !prdred_all_interp_values1

!
!
!
subroutine prdred(phia,la,ra,phib,lb,rb,rcen,lbdmxa,rhotb,rr,nr,jtb,clbdtb,lbdtb,nterm,xgla,wgla,ord,rho_min_jt,dr_jt) &
  bind(c, name='prdred_checkit')
  use m_fact, only : fac, sgn
  use m_interp_coeffs, only : interp_coeffs
  use m_interpolation, only : get_fval
!  use m_timing, only : get_cdatetime
  implicit none
  !! external
  integer(c_int), intent(in)  :: nr, nterm,la,lb,ord
  real(c_double), intent(in)  :: phia(nr),phib(nr),rr(nr),ra(3),rb(3),rcen(3)
  integer(c_int), intent(in)  :: lbdtb(nterm),clbdtb(nterm),lbdmxa,jtb(nterm)
  real(c_double), intent(in)  :: rho_min_jt, dr_jt
  real(c_double), intent(in)  :: xgla(ord), wgla(ord)
  real(c_double), intent(inout) :: rhotb(nr,nterm)
  
  !! internal
  real(8) :: ya(nr), yb(nr)
  real(8) :: raa,rbb,a1,a2,f1,f2,sumb,aa,bb,cc,thrj1,thrj2,r2aa,r2bb
  !real(8) :: t1,t2,tt(9)
  real(8) :: coeff1(-2:3), coeff2(-2:3)
  integer :: i,ix,ijmx,ij,clbd,kappa,kpmax,igla, lbd1_p_lbd2
  integer(c_int64_t) :: k, nr8
  integer :: lbd1,lbdp1,lbd2,lbdp2,lc,lcmin,lcmax,lcp,lcpmin,lcpmax,clbdp
  real(8), allocatable :: plval(:,:), fval(:,:)
  real(8) :: yz(ord)

!     write(6,*) 'prdred', lbdmxa, 2*lbdmxa+la+lb
  !_t1
  !tt = 0
  nr8 = int(nr, c_int64_t)
  allocate(plval(ord,0:2*lbdmxa+la+lb))
  allocate(fval(nr,0:2*lbdmxa+la+lb))
  
  kpmax = -999
  ijmx=la+lb;
  plval(:,0)=1.D0
  plval(:,1)=xgla(:)
  do kappa=1,2*lbdmxa+ijmx-1
    plval(:,kappa+1)=((2*kappa+1)*xgla*plval(:,kappa)-kappa*plval(:,kappa-1))/(kappa+1)
  end do
  ya=phia/rr**la
  yb=phib/rr**lb

  !_t2(tt(2))
  
  raa=sqrt(sum((ra-rcen)**2))
  rbb=sqrt(sum((rb-rcen)**2))
  fval=0
  do i=1,nr
    r2aa = rr(i)**2+raa**2
    r2bb = rr(i)**2+rbb**2
    
    do igla=1,ord
      a1=(r2aa-2.0d0*raa*rr(i)*xgla(igla))
      !call interp_coeffs(sqrt(a1), nr8, rho_min_jt, dr_jt, k, coeff1)
      !f1 = sum(ya(k:k+5)*coeff1)
      f1 = get_fval(ya, sqrt(a1), rho_min_jt, dr_jt, nr);

      a2=(r2bb+2.0d0*rbb*rr(i)*xgla(igla))
      !call interp_coeffs(sqrt(a2), nr8, rho_min_jt, dr_jt, k, coeff2)
      !f2 = sum(yb(k:k+5)*coeff2)
      f2 = get_fval(yb, sqrt(a2), rho_min_jt, dr_jt, nr);

      yz(igla)=f1*f2
    enddo
    kpmax=0
    if (raa+rbb .gt. 1.0d-5) kpmax=2*lbdmxa+ijmx
    do kappa=0,kpmax
      fval(i,kappa)=0.5d0*sum(plval(:,kappa)*yz*wgla)
    enddo
  enddo

  !_t2(tt(3))

  rhotb=0.0D0
  do ix=1,nterm
     ij=jtb(ix)
     clbd=clbdtb(ix)
     clbdp=lbdtb(ix)
     do lbd1=0,la
        lbdp1=la-lbd1
        aa=thrj(lbd1,lbdp1,la,0,0,0)*fac(lbd1)*fac(lbdp1)*fac(2*la+1)&
             /(fac(2*lbd1)*fac(2*lbdp1)*fac(la))

        do lbd2=0,lb
           lbdp2=lb-lbd2
           bb=thrj(lbd2,lbdp2,lb,0,0,0)*fac(lbd2)*fac(lbdp2)*fac(2*lb+1)&
                /(fac(2*lbd2)*fac(2*lbdp2)*fac(lb))
           bb=aa*bb
           do kappa=0,kpmax
              sumb=0.0d0
              lcmin=max(abs(lbd1-lbd2),abs(clbd-kappa))
              lcmax=min(lbd1+lbd2,clbd+kappa)
              do lc=lcmin,lcmax,2
                 thrj1 = thrj(lbd1,lbd2,lc,0,0,0)
                 thrj2 = thrj(lc,clbd,kappa,0,0,0)

                 lcpmin=max(abs(lbdp1-lbdp2),abs(clbdp-kappa))
                 lcpmax=min(lbdp1+lbdp2,clbdp+kappa)
                 do lcp=lcpmin,lcpmax,2
                    if ((abs(lc-ij).le.lcp).and.(lcp.le.lc+ij)) then
                       sumb=sumb+(2*lc+1)*(2*lcp+1)&
                            * thrj1 &
                            * thrj(lbdp1,lbdp2,lcp,0,0,0) &
                            * thrj2 &
                            * thrj(lcp,clbdp,kappa,0,0,0) &
                            * sixj(clbd,clbdp,ij,lcp,lc,kappa) &
                            * ninej(la,lb,ij,lbd1,lbd2,lc,lbdp1,lbdp2,lcp)
                    endif
                 enddo
              enddo
              cc=sgn(lbd1+kappa+lb)*(2*ij+1)*(2*kappa+1)&
                   *(2*clbd+1)*(2*clbdp+1)*bb*sumb
              if (cc .ne. 0.0D0) then
                lbd1_p_lbd2 = lbd1 + lbd2
                rhotb(:,ix)=rhotb(:,ix)+cc*rr(1:nr)**(lbd1_p_lbd2) &
                      * dpowi(raa, lbdp1) * dpowi(rbb, lbdp2)*fval(:,kappa)
              endif
           enddo
        enddo
     enddo
  enddo
  
  !_t2(tt(4))
  !write(6,*) __FILE__, __LINE__, tt(1:4)
 
  _dealloc(fval)
  _dealloc(plval)
  
  return
end subroutine !prdred


!
!
!
pure real(8) function dpowi(a,b)
  implicit none
  real(8), intent(in) :: a
  integer, intent(in) :: b
  if(b==0) then
    dpowi = 1
  else
    dpowi = a**b
  endif
end function ! dpowi

!
!
!
subroutine thrj_subr(l1,l2,l3,m1,m2,m3,thrjres) bind(c, name='thrj_subr')
  implicit none
  integer(c_int), intent(in) :: l1,l2,l3,m1,m2,m3   !! exernal
  real(c_double), intent(inout) :: thrjres
  thrjres = thrj(l1,l2,l3,m1,m2,m3)
end subroutine ! thrj_subr

!
! 3j, 6j, 9j symbols from Talman
!
pure function thrj(l1,l2,l3,m1,m2,m3)
  use m_fact, only : sgn, fac
  implicit none
  integer(c_int), intent(in) :: l1,l2,l3,m1,m2,m3   !! exernal
  real(c_double) :: thrj

  !! internal
  real(c_double) :: xx,cc,s,ph,t,dlt
  integer(c_int) :: lg,it,lgh,itmin,itmax
  thrj=0.0D0 
  if (m1**2.gt.l1**2) return 
  if (m2**2.gt.l2**2) return 
  if (m3**2.gt.l3**2) return 
  if (m1+m2+m3.ne.0) return 
  if (l3.lt.iabs(l1-l2)) return 
  if (l3.gt.l1+l2) return 
  lg=l1+l2+l3
  dlt=sqrt(fac(lg-2*l1)*fac(lg-2*l2)*fac(lg-2*l3)/fac(lg+1))
  if ((m1.eq.0).and.(m2.eq.0)) then
    if (mod(lg,2).eq.1) return
    lgh=lg/2
    thrj=sgn(lgh)*dlt*fac(lgh)/(fac(lgh-l1)*fac(lgh-l2)*fac(lgh-l3))
    return
  endif
  xx=fac(l3+m3)*fac(l3-m3)/(fac(l1+m1)*fac(l1-m1)&
       *fac(l2+m2)*fac(l2-m2)) 
  cc=dlt*sqrt(xx)
  itmin=max0(0,l1-l2+m3) 
  itmax=min0(l3-l2+l1,l3+m3) 
  s=0.0D0
  ph=1.0D0 
  it=itmin 
  if (mod(it+l2+m2,2).eq.1) ph=-ph 
  12 t=ph*fac(l3+l1-m2-it)*fac(l2+m2+it) 
  t=t/(fac(l3+m3-it)*fac(it+l2-l1-m3)*fac(it)*fac(l3-l2+l1-it)) 
  s=s+t 
  it=it+1 
  ph=-ph 
  if (it.le.itmax) go to 12 
  thrj=cc*s 
  return 
end function !thrj


pure  function ninej(j1,j2,j12,j3,j4,j34,j13,j24,j)
  implicit none
  !! external
  integer(c_int), intent(in) :: j1,j2,j12,j3,j4,j34,j13,j24,j
  real(8) :: ninej
  !! internal
  integer ::  x,xmin,xmax
  real(8) :: ss
  xmin=max(abs(j1-j),abs(j2-j34),abs(j3-j24))
  xmax=min(j1+j,j2+j34,j3+j24)
  ss=0.0D0
  do x=xmin,xmax
     ss=ss+(2*x+1)*sixj(j1,j2,j12,j34,j,x)*sixj(j3,j4,j34,j2,x,j24)&
          *sixj(j13,j24,j,x,j1,j3)
  enddo
  ninej=ss
  return
end function !ninej


pure  function sixj(j1,j2,j3,l1,l2,l3)
  use m_fact, only : sgn, fac
  implicit none
  !! external
  integer, intent(in) :: j1,j2,j3,l1,l2,l3
  real(8) :: sixj
  !! internal
  integer :: z,zmin,zmax
  real(8) :: aa,ab,ac,ad,cc,ss
  
  aa=fac(j1+j2-j3)*fac(j2+j3-j1)*fac(j3+j1-j2)/fac(j1+j2+j3+1)
  ab=fac(j1+l2-l3)*fac(l3+j1-l2)*fac(l2+l3-j1)/fac(j1+l2+l3+1)
  ac=fac(j2+l3-l1)*fac(l1+j2-l3)*fac(l3+l1-j2)/fac(j2+l1+l3+1)
  ad=fac(j3+l1-l2)*fac(l2+j3-l1)*fac(l1+l2-j3)/fac(j3+l1+l2+1)
  cc=sqrt(aa*ab*ac*ad)
  zmax=min(j1+j2+l1+l2,j1+j3+l1+l3,j2+j3+l2+l3)
  zmin=max(j1+j2+j3,j1+l2+l3,l1+j2+l3,l1+l2+j3)
  ss=0.0D0
  do z=zmin,zmax
     ss=ss+sgn(z)*fac(z+1)/(fac(z-j1-j2-j3)*fac(z-j1-l2-l3)*fac(z-j2-l1-l3)&
          *fac(z-j3-l1-l2)*fac(j1+j2+l1+l2-z)*fac(j1+j3+l1+l3-z)&
          *fac(j2+j3+l2+l3-z))
  enddo
  sixj=cc*ss
  return
end function !sixj

!
!
!
subroutine csphar_talman(r,ylm,lmax) bind(c, name='csphar_talman')
  use m_fact, only : sgn
  implicit none 
  real(c_double), intent(in) :: r(3)
  integer(c_int), intent(in) :: lmax
  complex(c_double_complex), intent(inout) :: ylm(*)
  
  real(c_double) :: x,y,z,dd,phi,cc,ss,al,aa,bb,zz,cs
  integer ll,l,m,il1,il2,ind,ll2
  real(c_double) :: pi, rttwo
  pi = 4D0*atan(1D0)
  rttwo=sqrt(2.0D0)

  x=r(1) 
  y=r(2) 
  z=r(3) 
  dd=sqrt(x*x+y*y+z*z)
  if (dd.lt.1.0d-10) then
     ll=(lmax+1)**2-1
     do  l=1,ll 
       ylm(l+1)=0.0D0 
     end do
     ylm(1)=1.0D0
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
  ylm(1)=1.0D0
  if (lmax.eq.0) return
  do l=1,lmax 
    al=l 
    il2=(l+1)**2-1 
    il1=l**2-1
    ylm(il2+1)=-ss*sqrt((al-0.5D0)/al)*ylm(il1+1) 
    ylm(il2)=cc*sqrt(2.0D0*al-1.0D0)*ylm(il1+1)
  end do
  if (lmax.ge.2) then
    do m=0,lmax-2
      if (m.lt.lmax) then
        do l=m+1,lmax-1
          ind=l*(l+1)+m 
          aa=l**2-m**2
          bb=(l+1)**2-m**2
          zz=(l+l+1.0D0)*cc*dble(ylm(ind+1))-sqrt(aa)*dble(ylm(ind-2*l+1)) 
          ylm(ind+2*(l+1)+1)=zz/sqrt(bb) 
        end do
      endif
    enddo
  endif
  do l=0,lmax
    ll2=l*(l+1)
    do m=0,l
      cs=sin(m*phi)
      cc=cos(m*phi)
      ylm(ll2+m+1)=cmplx(cc,cs,8)*ylm(ll2+m+1)
      ylm(ll2-m+1)=sgn(m)*conjg(ylm(ll2+m+1))
    enddo
  enddo
  return 

end subroutine !csphar_talman

!!
!!
!!
!subroutine gauleg(x,w,n)
!  implicit none
!  integer n
!  real(8) :: eps,pi,z,p1,p2,p3,pp,z1,x(:),w(:)
!  integer i,m,j
!  eps=3.0D-14
!  m=(n+1)/2
!  pi=4.0D0*atan(1.0d0)
!  do i=1,m
!     z=cos(pi*(i-0.25D0)/(n+0.5D0))
!1    continue
!     p1=1.0D0
!     p2=0.0D0
!     do j=1,n
!       p3=p2
!       p2=p1
!       p1=((2.0D0*j-1)*z*p2-(j-1)*p3)/j
!     enddo
!     pp=n*(z*p1-p2)/(z*z-1.0D0)
!     z1=z
!     z=z1-p1/pp
!     if (abs(z-z1).gt.eps) go to 1
!     x(i)=-z
!     x(n+1-i)=z
!     w(i)=2.0D0/((1.0D0-z*z)*pp*pp)
!     w(n+1-i)=w(i)
!  enddo
!  return
!end subroutine !gauleg


!!
!! Shift of radial orbital ?
!!
!subroutine trn(nr,ar,dr,rhomin,ngl,xgl,wgl, f,rc,li,tx,clz,clzp,nt,lcmax,dim1)
!  use m_fact, only : fac, sgn
!  use m_interpolation, only : get_fval

!  implicit none
!  !! external
!  integer, intent(in)    :: nr
!  real(8), intent(in)    :: ar(nr)
!  real(8), intent(in)    :: rhomin,dr
!  integer, intent(in)    :: ngl
!  real(8), intent(in)    :: xgl(ngl),wgl(ngl)
!  integer, intent(in)    :: dim1,li,lcmax
!  integer, intent(inout) :: nt,clz(dim1),clzp(dim1) 
!  real(8), intent(out)   :: tx(nr,dim1)
!  real(8), intent(in)    :: f(nr), rc
  
!  !! internal
!  integer :: lbdmin,lbdmax,ll,i,j,lbd,lc2,ic,lc1,lc2min,lc2max,ll1,ll2
!  real(8) :: rmn,rmx,xx,yy,zz,aa,qq1,qq2,qq3
!  real(8) :: ya(nr),yb(ngl)
!  real(8), allocatable :: ta(:,:), tb(:,:), pltb(:,:)
  
!  allocate( ta(nr, 0:50), tb(nr, 0:50), pltb(ngl, 0:50) )
  
!  lbdmax=lcmax+li
!  if (rc.lt.1.0d-5) then
!    nt=1
!    clz=0
!    clzp=0
!    clz(1)=li
!    tx=0
!    tx(:,1)=(2*li+1)*f
!    return
!  endif
!  ya=f/ar(1:nr)**li
!  ta(:,0)=rc**li
!  if (li.gt.0) then
!    do ll=1,li
!      ta(:,ll)=ar(1:nr)*ta(:,ll-1)/rc
!    enddo
!  endif

!  do i=1,nr
!    rmn=min(rc,ar(i))
!    rmx=max(rc,ar(i))
!    do j=1,ngl
!      xx=rmx+xgl(j)*rmn
!      yy = get_fval(ya,xx,rhomin,dr,nr)
!      yb(j)=xx*yy
!      zz=-xgl(j)+0.5d0*rmn*(1.0d0-xgl(j)**2)/rmx
!      pltb(j,0)=1.0D0
!      pltb(j,1)=zz
!      do ll=1,lbdmax-1
!        pltb(j,ll+1)=((2*ll+1)*zz*pltb(j,ll)-ll*pltb(j,ll-1))/(ll+1)
!      enddo
!    enddo
!    do lbd=0,lbdmax
!      tb(i,lbd)=0.5d0*(2*lbd+1)*sum(wgl(1:ngl)*pltb(:,lbd)*yb)/rmx
!    enddo
!  enddo

!  tx=0
!  ic=0

!  do lc1=0,lcmax
!     lc2min=abs(lc1-li)
!     lc2max=lc1+li
!     do lc2=lc2min,lc2max,2
!        ic=ic+1
!        do ll1=0,li
!           ll2=li-ll1
!           lbdmin=max(abs(lc1-ll1),abs(lc2-ll2))
!           lbdmax=min(lc1+ll1,lc2+ll2)
!           if (lbdmin.le.lbdmax) then
!              do lbd=lbdmin,lbdmax,2
!                 qq1=(fac((lc1+lbd+ll1)/2)*fac(lc1+lbd-ll1))&
!                      /(fac((lc1+lbd-ll1)/2)*fac((ll1+lbd-lc1)/2)&
!                      *fac((lc1+ll1-lbd)/2)*fac(lc1+lbd+ll1+1))
!                 qq3=(fac((lc1+lc2+li)/2)*fac(lc1+lc2-li))&
!                      /(fac((lc1+lc2-li)/2)*fac((lc1+li-lc2)/2)&
!                      *fac((lc2+li-lc1)/2)*fac(lc1+lc2+li+1))
!                 qq2=(fac((lc2+lbd+ll2)/2)*fac(lc2+lbd-ll2))&
!                      /(fac((lc2+lbd-ll2)/2)*fac((lc2+ll2-lbd)/2)&
!                      *fac((ll2+lbd-lc2)/2)*fac(lc2+lbd+ll2+1))
!                 aa=sgn(ll2)*(2*lc1+1)*(2*lc2+1)*qq1*qq2/qq3

!                 if (aa.ne.0.0) then
!                    clz(ic)=lc1
!                    clzp(ic)=lc2
!                    tx(:,ic)=tx(:,ic)+aa*ta(:,ll1)*tb(:,lbd)
!                 endif
!              enddo

!           endif
!        enddo
!     enddo
!  enddo

!  nt=ic
  
!end subroutine !trn


end module m_prod_talman
