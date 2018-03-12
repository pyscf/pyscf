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

module m_functs_l_mult_type
!
! The purpose of the module is to store and deal with a functions in m-multipletts
!
#include "m_define_macro.F90"
  use m_die, only : die

  implicit none
  private die

  !! Descr of real space information, local dominant products, atomic orbitals...
  type functs_l_mult_t
    real(8), allocatable :: ir_mu2v(:,:) ! collection of radial functions
    integer, allocatable :: mu2j(:)    ! multiplett -> j    (allocated for l-multipletts i.e. local dp)
    integer, allocatable :: mu2si(:)   ! multiplett -> start index in local counting (allocated for l-multipletts i.e. local dp)
    real(8), allocatable :: mu2rcut(:) ! cutoffs per radial orbital 
    real(8) :: rcut           =-999    ! Spatial cutoff of the function
    integer :: sp             =-999    ! atom specie
    integer :: nfunct         =-999    ! number of functions in this specie
  end type ! functs_l_mult_t
  !! END of Descr of real space information, local dominant products, atomic orbitals...

  contains


!
!
!
function get_spent_ram(lsp) result(ram_bytes)
  use m_get_sizeof
  implicit none
  type(functs_l_mult_t), intent(in) :: lsp

  real(8) :: ram_bytes
  integer(8) :: nr, ni

  nr = 0
  _add_size_alloc(nr, lsp%ir_mu2v)
  _add_size_alloc(nr, lsp%mu2rcut)
  
  ni = 0
  _add_size_alloc(ni, lsp%mu2j)
  _add_size_alloc(ni, lsp%mu2si)
  ni = ni + 1
  ram_bytes = nr*get_sizeof(lsp%mu2rcut(1)) + ni*get_sizeof(lsp%mu2si(1))

end function !get_spent_ram

!
!
!
subroutine init_functs_l_mult(nmu, ir_mu2v, mu2j, mu2rcut, f, sp)
  implicit none
  !! external
  integer, intent(in) :: nmu, mu2j(:)
  real(8), intent(in) :: ir_mu2v(:,:)
  real(8), intent(in) :: mu2rcut(:)
  type(functs_l_mult_t), intent(inout) :: f
  integer, intent(in), optional :: sp
  !! internal
  integer :: nr, nn(3), mm(2), si, fi, nf, mu, j
  nr = size(ir_mu2v,1)
  if(nr<1) _die('!size(ir_mu2v,1)<1')
  nn(1) = size(ir_mu2v,2)
  nn(2) = size(mu2j)
  nn(3) = size(mu2rcut)
  if(any(nn<nmu)) _die('!nmu ? ir_mu2v ? mu2j ? mu2rcut')
  mm = [nr, nmu]
_dealloc_u(f%ir_mu2v, mm)
  if(.not. allocated(f%ir_mu2v)) allocate(f%ir_mu2v(mm(1),mm(2)))
_dealloc_u(f%mu2j, nmu)
  if(.not. allocated(f%mu2j)) allocate(f%mu2j(nmu))
_dealloc_u(f%mu2si, nmu)
  if(.not. allocated(f%mu2si)) allocate(f%mu2si(nmu))
_dealloc_u(f%mu2rcut, nmu)
  if(.not. allocated(f%mu2rcut)) allocate(f%mu2rcut(nmu))

  f%ir_mu2v(1:nr, 1:nmu) = ir_mu2v(1:nr,1:nmu)
  f%mu2j(1:nmu) = mu2j(1:nmu)
  f%mu2rcut(1:nmu) = mu2rcut(1:nmu)
  f%rcut = maxval(f%mu2rcut)
  f%sp = -1
  if(present(sp)) f%sp = sp
  
  fi = 0
  f%nfunct = 0
  do mu=1,nmu
    j = get_j(f, mu)
    nf = 2*j+1
    si = fi + 1; fi = si + nf - 1
    f%mu2si(mu) = si
    f%nfunct = f%nfunct + nf
  enddo ! mu
  
  
end subroutine ! init_functs_l_mult  
  
!
!
!
real(8) function get_rcut(lmult)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: lmult
  !! internal
  if(lmult%rcut<0) _die('lmult%rcut<0 ??')
  get_rcut = lmult%rcut
  
end function ! get_rcut

!
!
!
real(8) function get_radii(ff, rr, dlambda)
  implicit none
  !! external
  real(8), intent(in) :: ff(:), rr(:)
  real(8), intent(in) :: dlambda
  !! internal
  get_radii = sqrt(sum(ff**2*(rr**5))*dlambda)
  
end function ! get_radii

!
!
!
real(8) function get_radii_sqr(ff, rr, dlambda)
  implicit none
  !! external
  real(8), intent(in) :: ff(:), rr(:)
  real(8), intent(in) :: dlambda
  !! internal
  get_radii_sqr = sum(ff**2*(rr**5))*dlambda

end function ! get_radii_sqr

!
!
!
real(8) function get_mean_radii(lmult, rr)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: lmult
  real(8), intent(in) :: rr(:)
  !! internal
  integer :: j, nmu, nr, nf, mu, iw
  real(8) :: dlambda
  real(8), pointer :: ff(:)
  nr = get_nr_lmult(lmult)
    
  dlambda = log(rr(nr)/rr(1))/(nr-1)
  nmu = get_nmult(lmult)
  get_mean_radii = 0
  nf = 0
  do mu=1,nmu
    j = get_j(lmult, mu)
    iw = (2*j+1)
    ff=>get_ff_ptr(lmult, mu)
    get_mean_radii = get_mean_radii + sqrt(sum(ff**2*(rr**5))*dlambda)*iw / 2
    nf = nf + iw
  enddo ! 
  get_mean_radii = get_mean_radii / nf
  
end function ! get_radii


!
! Computes Fourier transform of l-multipletts for a set of momenta pvec(1:3,1:nmomenta)
!
subroutine comp_ft_lmult(pp, sp2functs_mom, nmomenta, pvecs, sp2ft_functs)
  use m_arrays, only : z_array2_t
  use m_interpolation, only : grid_2_dr_and_rho_min, get_fval
  use m_harmonics, only : rsphar
  implicit none
  !! external
  real(8), intent(in) :: pp(:)
  type(functs_l_mult_t), intent(in), allocatable :: sp2functs_mom(:)
  integer, intent(in) :: nmomenta
  real(8), intent(in) :: pvecs(:,:)
  type(z_array2_t), intent(inout), allocatable :: sp2ft_functs(:)
  !! internal
  integer :: ip, nsp, sp, nr, nf, nmult, mu, j, si, fi, jmax
  real(8) :: pvec(3), psca, dp_jt, kmin_jt, fp_mom, pi
  real(8), allocatable :: ff(:), slm(:)
  complex(8) :: zi
  pi = 4D0*atan(1D0);
  zi = cmplx(0D0, 1D0,8);
  
  _dealloc(sp2ft_functs)
  nsp = 0
  if(allocated(sp2functs_mom)) nsp = size(sp2functs_mom)
  if (nsp<1) return;

  if(3>size(pvecs,1)) _die('3>size(pvecs,1)')
  if(nmomenta>size(pvecs,2)) _die('nmomenta>size(pvecs,2)')
  if(.not. allocated(sp2functs_mom)) _die('.not. allocated(sp2functs_mom)')
  nr = get_nr_lmult(sp2functs_mom(1))
  jmax = get_jcutoff_lmult(sp2functs_mom)
  
  allocate(sp2ft_functs(nsp))
  
  call grid_2_dr_and_rho_min(nr, pp, dp_jt, kmin_jt)

  do sp=1,nsp
    nf = get_nfunct_lmult(sp2functs_mom(sp))
    allocate(sp2ft_functs(sp)%array(nmomenta,nf))
    sp2ft_functs(sp)%array = 0
  enddo ! sp  

  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP SHARED(nr,nmomenta,sp2ft_functs,jmax,nsp,sp2functs_mom,kmin_jt,dp_jt,zi) &
  !$OMP SHARED(pvecs) &
  !$OMP PRIVATE(pvec,psca,slm,ff,nf,sp,mu,nmult,j,si,fi,fp_mom,ip)  
  allocate(ff(nr))
  allocate(slm(0:(jmax+1)**2))

  !! Loop over momenta for which we should compute Fourier transform  
  !$OMP DO
  do ip=1, nmomenta
    pvec = pvecs(:,ip)
    psca = sqrt(sum(pvec**2))
    call rsphar(pvec, slm(0:), jmax)

    !! Loop over (l-multiplett) species
    do sp=1,nsp
      nf = get_nfunct_lmult(sp2functs_mom(sp))
      nmult = get_nmult(sp2functs_mom(sp))
      do mu=1,nmult
        call get_j_si_fi(sp2functs_mom(sp), mu, j, si, fi)
        call get_ff_lmult(sp2functs_mom(sp), mu, ff)
        fp_mom = get_fval(ff, psca, kmin_jt, dp_jt, nr)
        
        sp2ft_functs(sp)%array(ip,si:fi) = &
          conjg((zi**j)) * fp_mom * slm( j*(j+1)-j:j*(j+1)+j )
      enddo ! mu=1,nmult
    enddo ! sp=1,nsp
    !! END of Loop over (l-multiplett) species
  enddo
  !$OMP END DO
  !! END of Loop over momenta for which we should compute Fourier transform
  _dealloc(ff)
  _dealloc(slm)
  !$OMP END PARALLEL

  do sp=1,nsp
    sp2ft_functs(sp)%array = sp2ft_functs(sp)%array * sqrt(pi/2)*(4*pi)
  enddo ! sp

end subroutine ! comp_ft_lmult


!
!
!
subroutine init_functs_lmult_mom_space(Talman_plan, sp2functs_rea, sp2functs_mom)
  use m_sph_bes_trans, only : sbt_execute, Talman_plan_t, get_nr
  implicit none
  !! external
  type(Talman_plan_t), intent(in) :: Talman_plan
  type(functs_l_mult_t), intent(in), allocatable :: sp2functs_rea(:)
  type(functs_l_mult_t), intent(inout), allocatable :: sp2functs_mom(:)
  !! internal
  integer :: j,spp,nmult,si,fi,mu,nspp,nr
  real(8), allocatable :: ff(:)

  nr = get_nr(Talman_plan)
  allocate(ff(nr))
  
  _dealloc(sp2functs_mom)
  if(.not. allocated(sp2functs_rea)) return
  nspp = size(sp2functs_rea)
  allocate(sp2functs_mom(nspp))

  do spp=1,nspp
    sp2functs_mom(spp) = sp2functs_rea(spp)
    nmult = get_nmult(sp2functs_rea(spp))
    do mu=1,nmult
      call get_j_si_fi(sp2functs_rea(spp), mu, j, si, fi)
      call get_ff_lmult(sp2functs_rea(spp), mu, ff)
      call sbt_execute(Talman_plan, ff, sp2functs_mom(spp)%ir_mu2v(:,mu), j, 1)
    enddo ! mu=1,nmult
  enddo ! spp
  
  _dealloc(ff)

end subroutine ! init_functs_mmult_mom_space

!
!
!
subroutine init_moms_lmult(sp2functs_rea, rr, sp2moms)
  use m_interpolation, only : get_dr_jt
  implicit none
  !! external
  type(functs_l_mult_t), intent(in), allocatable :: sp2functs_rea(:)
  real(8), intent(in) :: rr(:)
  type(functs_l_mult_t), intent(inout), allocatable :: sp2moms(:)
  !! internal
  integer :: j,spp,nmult,mu,si,fi
  real(8) :: dlambda
  !! Dimensions
  integer :: nspp, nr
  !! END of Dimensions

  _dealloc(sp2moms)
  if(.not. allocated(sp2functs_rea)) return
  nspp = size(sp2functs_rea)
  allocate(sp2moms(nspp))
  nr = get_nr_lmult(sp2functs_rea(1))
  dlambda = get_dr_jt(rr)

  do spp=1,nspp
    sp2moms(spp) = sp2functs_rea(spp)
    nmult = get_nmult(sp2functs_rea(spp))
    _dealloc(sp2moms(spp)%ir_mu2v)
    allocate(sp2moms(spp)%ir_mu2v(1,nmult))
    do mu=1,nmult
      call get_j_si_fi(sp2functs_rea(spp), mu, j, si, fi)
      sp2moms(spp)%ir_mu2v(1,mu) = dlambda * &
          sum(sp2functs_rea(spp)%ir_mu2v(1:nr,mu)*rr**(j+3))
    enddo ! mu
  enddo ! spp

end subroutine ! init_moms_lmult

!
! Counts number of functions in an l-multiplett specie
!
function get_nfunct_lmult(functs) result (nf)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs
  integer :: nf
  !! internal
  integer :: nmult, mu, nf1, nf2, nf3, si, fi, j
  nf = 0
  if(.not. allocated(functs%ir_mu2v)) _die('ir_mu2v ?')
  if(.not. allocated(functs%mu2j)) _die('mu2j ?')
  if(.not. allocated(functs%mu2si)) _die('mu2si ?')

  nf1 = functs%nfunct

  nmult = get_nmult(functs)
  nf2 = 0
  do mu=1,nmult
    call get_j_si_fi(functs, mu, j, si, fi)
    nf2 = nf2 + (2*j+1)
  enddo ! mu=1,nmult

  call get_j_si_fi(functs, nmult, j, si, fi)
  nf3 = si + 2*j

  nf = nf1

  if(nf1/=nf2) _die('nf1/=nf2')
  if(nf2/=nf3) then
    write(0,'(a,2i8)') 'si    j ', si, j
    write(0,'(a,2i8)') 'nf2  nf3', nf2, nf3
    _die('nf2/=nf3')
  endif

end function ! get_nfunct_lmult

!
! Counts number of multipletts in l-specie
!
function get_nmult(functs) result (nf)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs
  integer :: nf
  !! internal
  integer :: nn(4)
  nf = 0
  if(.not. allocated(functs%mu2rcut)) _die('mu2rcut ?')
  if(.not. allocated(functs%ir_mu2v)) _die('ir_mu2v ?')
  if(.not. allocated(functs%mu2j)) _die('mu2j ?')
  if(.not. allocated(functs%mu2si)) _die('mu2si ?')

  nn(1) = size(functs%ir_mu2v,2)
  nn(2) = size(functs%mu2j)
  nn(3) = size(functs%mu2si)
  nn(4) = size(functs%mu2rcut)
  if(any(nn(1)/=nn)) _die('nmult ?')

  nf = nn(1)
  if(nf<1) _die('nmult <1')

end function ! get_nmult

!
! Returns angular momentum, start and finish indices for a multiplett in l-specie
!
subroutine get_j_si_fi(functs, mu, j, si, fi)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs
  integer, intent(in) :: mu
  integer, intent(out) :: j, si, fi
  !! internal
  if(.not. allocated(functs%ir_mu2v)) _die('ir_mu2v ?')
  if(.not. allocated(functs%mu2j)) _die('mu2j ?')
  if(.not. allocated(functs%mu2si)) _die('mu2si ?')

  if(mu>size(functs%mu2j)) _die('mu>size(functs%mu2j)')
  if(mu>size(functs%mu2si)) _die('mu>size(functs%mu2si)')
  if(mu<1) _die('mu<1')
  j = functs%mu2j(mu)
  si = functs%mu2si(mu)
  fi = si + 2*j

end subroutine ! get_j_si_fi

!
! Returns radial function of a given multiplett
!
subroutine get_ff_lmult(functs, mu, ff)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs
  integer, intent(in) :: mu
  real(8), intent(out) :: ff(:)
  !! internal
  if(.not. allocated(functs%ir_mu2v)) _die('ir_mu2v ?')
  if(.not. allocated(functs%mu2j)) _die('mu2j ?')
  if(.not. allocated(functs%mu2si)) _die('mu2si ?')

  if(mu>size(functs%ir_mu2v,2)) _die('mu>size(functs%ir_mu2v)')
  if(mu>size(functs%mu2j)) _die('mu>size(functs%mu2j)')
  if(mu>size(functs%mu2si)) _die('mu>size(functs%mu2si)')
  if(mu<1) _die('mu<1')
  ff(:) = functs%ir_mu2v(:,mu)

end subroutine ! get_ff_lmult

!
! Returns radial function of a give multiplett
!
function get_ff_ptr(functs, mu) result(ptr)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in), target :: functs
  integer, intent(in) :: mu
  real(8), pointer :: ptr(:)
  !! internal
  if(.not. allocated(functs%ir_mu2v)) _die('ir_mu2v ?')
  if(mu>size(functs%ir_mu2v,2)) _die('mu>size(functs%ir_mu2v)')
  if(mu<1) _die('mu<1')
  ptr => functs%ir_mu2v(:,mu)

end function ! get_ff_ptr

!
! Returns angular momentum, start and finish indices for a multiplett in l-specie
!
integer function get_j(functs, mu)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs
  integer, intent(in) :: mu
  !! internal
  if(.not. allocated(functs%mu2j)) _die('mu2j ?')
  if(mu>size(functs%mu2j)) _die('mu>size(functs%mu2j)')
  if(mu<1) _die('mu<1')
  if(functs%mu2j(mu)<0) _die('mu2j(mu)<0')
  get_j = functs%mu2j(mu)
end function ! get_j

!
! Returns number of points of the radial grid
!
function get_nr_lmult(functs) result (nf)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs
  integer :: nf
  !! internal
  nf = 0
  if(.not. allocated(functs%ir_mu2v)) _die('ir_mu2v ?')

  nf = size(functs%ir_mu2v,1)

end function ! get_nr_lmult

!
! Computes scalar moments of the functions stored in l-multipletts
!
subroutine comp_sca_mom_lmult(rr, sp_local2functs, sp_local2sca_mom)
  use m_arrays, only : d_array1_t
  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_l_mult_t), intent(in), allocatable :: sp_local2functs(:)
  type(d_array1_t), intent(inout), allocatable :: sp_local2sca_mom(:)
  !! internal
  integer :: nsp, sp, j, si, fi, mu, nmult, nr
  real(8) :: pi, d_lambda
  real(8), pointer :: ff(:)
  pi = 4D0*atan(1D0)
  nsp = 0
  if(allocated(sp_local2functs)) nsp = size(sp_local2functs)
  if (nsp<1) then; _dealloc(sp_local2sca_mom); return; endif

  allocate(sp_local2sca_mom(nsp))
  nr = get_nr_lmult(sp_local2functs(1))
  d_lambda=log(rr(nr)/rr(1))/(nr-1)

  !! Loop over (l-multiplett) species
  do sp=1,nsp
    allocate(sp_local2sca_mom(sp)%array(get_nfunct_lmult(sp_local2functs(sp))))
    sp_local2sca_mom(sp)%array = 0
    nmult = get_nmult(sp_local2functs(sp))
    do mu=1,nmult
      call get_j_si_fi(sp_local2functs(sp), mu, j, si, fi)
      if(j>0) cycle
      ff => get_ff_ptr(sp_local2functs(sp), mu)
      sp_local2sca_mom(sp)%array(si:fi) = sqrt(4*pi)*d_lambda*sum(ff*rr**3)
    enddo ! mu=1,nmult
    !write(6,*)sp, sp_local2sca_mom(sp)%array
  enddo ! sp=1,nsp
  !! END of Loop over (l-multiplett) species

end subroutine ! comp_sca_mom_lmult

!
! Computes scalar moments of the functions stored in l-multipletts
!
subroutine comp_norm_lmult(rr, functs, norms)
  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_l_mult_t), intent(in) :: functs
  real(8), intent(inout), allocatable :: norms(:)
  !! internal
  integer :: j, si, fi, mu, nmult, nr, nf
  real(8) :: pi, d_lambda
  real(8), pointer :: ff(:)
  pi = 4D0*atan(1D0)
  nr = get_nr_lmult(functs)
  d_lambda=log(rr(nr)/rr(1))/(nr-1)

  nf = get_nfunct_lmult(functs)
  if(nf<1) _die('!nf')
  
  if(allocated(norms)) then
    if(size(norms)<nf) deallocate(norms)
  endif
  if(.not. allocated(norms)) allocate(norms(nf))
  norms = 0
  
  nmult = get_nmult(functs)
  do mu=1,nmult
    call get_j_si_fi(functs, mu, j, si, fi)
    ff => get_ff_ptr(functs, mu)
    norms(si:fi) = sqrt(4*pi*d_lambda**2 * sum(ff**2*rr**3))
  enddo ! mu=1,nmult

end subroutine ! comp_norm_lmult

!
! Computes dipole moments of the functions stored in l-multipletts
! Coordinates of the origin are zero
!
subroutine comp_dip_mom_lmult(rr, sp2functs, sp2dip_mom)
  use m_arrays, only : d_array2_t
  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_l_mult_t), intent(in), allocatable :: sp2functs(:)
  type(d_array2_t), intent(inout), allocatable :: sp2dip_mom(:) ! sp%array(funct,xyz)
  !! internal
  integer :: nsp, sp, j, si, fi, mu, nmult, nr
  real(8) :: pi, d_lambda, dip_mom(3,-1:1), frr
  real(8), allocatable :: ff(:)
  pi = 4D0*atan(1D0)
  nsp = 0
  if(allocated(sp2functs)) nsp = size(sp2functs)
  if (nsp<1) then; _dealloc(sp2dip_mom); return; endif

  allocate(sp2dip_mom(nsp))
  nr = get_nr_lmult(sp2functs(1))
  allocate(ff(nr))
  _zero(ff)
  d_lambda=log(rr(nr)/rr(1))/(nr-1)

  !! Loop over (l-multiplett) species
  do sp=1,nsp
    allocate(sp2dip_mom(sp)%array(get_nfunct_lmult(sp2functs(sp)),3))
    sp2dip_mom(sp)%array = 0
    nmult = get_nmult(sp2functs(sp))
    do mu=1,nmult
      call get_j_si_fi(sp2functs(sp), mu, j, si, fi )
      if(j/=1) cycle
      call get_ff_lmult(sp2functs(sp), mu, ff )
      frr = sqrt(4*pi/3D0)*d_lambda*sum(ff*rr**4)
      dip_mom = 0
      dip_mom(2,-1) = frr
      dip_mom(3, 0) = frr
      dip_mom(1, 1) = frr
      sp2dip_mom(sp)%array(si:fi,1:3) = transpose(dip_mom(1:3,-1:1))
    enddo ! mu=1,nmult
    !write(6,*)sp, sp_local2sca_mom(sp)%array
  enddo ! sp=1,nsp
  !! END of Loop over (l-multiplett) species

end subroutine ! comp_dip_mom_lmult

!
!
!
function get_jcutoff_lmult(sp2functs) result(jcutoff)
  implicit none
  type(functs_l_mult_t), intent(in), allocatable :: sp2functs(:)
  integer :: jcutoff
  !! internal
  integer :: spp, nspp, mu, nmult, j, si, fi
  
  jcutoff = -1;
  nspp = 0; if(allocated(sp2functs)) nspp = size(sp2functs)
  do spp=1,nspp
    nmult = get_nmult(sp2functs(spp))
    do mu=1,nmult
      call get_j_si_fi(sp2functs(spp), mu, j, si, fi)
      jcutoff = max(jcutoff, j)
    enddo ! mu  
  enddo ! spp

end function !get_jcutoff_lmult

!
!
!
function get_jcut_lmult(sp2functs) result(jcutoff)
  implicit none
  type(functs_l_mult_t), intent(in) :: sp2functs
  integer :: jcutoff
  !! internal
  integer :: mu, nmult, j, si, fi
  
  jcutoff = -1;
  nmult = get_nmult(sp2functs)
  do mu=1,nmult
    call get_j_si_fi(sp2functs, mu, j, si, fi)
    jcutoff = max(jcutoff, j)
  enddo ! mu  

end function !get_jcut_lmult

!
! Computes a mean radii for each multiplett
!
subroutine comp_mu_sp2radii(psi_log, sp2nmult, mu_sp2j, rr, mu_sp2radii)
  !!  external
  implicit none
  real(8), intent(in) :: psi_log(:,:,:)
  integer, intent(in) :: sp2nmult(:), mu_sp2j(:,:)
  real(8), intent(in) :: rr(:)
  real(8), intent(out) :: mu_sp2radii(:,:)
  !! internal
  integer :: sp, mu, j
  real(8) :: dlambda, radii
  !! Dimensions
  integer :: nr, nspecies
  nr = size(rr)
  nspecies = size(psi_log,3)
  dlambda = log(rr(nr)/rr(1))/(nr-1)
  
  mu_sp2radii = -999
  do sp=1, nspecies
    do mu=1,sp2nmult(sp)
      j = mu_sp2j(mu, sp)
      radii = get_radii(psi_log(:,mu,sp), rr, dlambda)
      mu_sp2radii(mu, sp) = radii * (2*j+1D0)
    end do
  end do
end subroutine ! comp_mu_sp2radii


!
! Maxiaml number of multipletts (pseudo-atomic orbitals) in l-species
!
function get_nmult_max(functs) result(nf)
  implicit none
  !! external
  type(functs_l_mult_t), intent(in) :: functs(:)
  integer :: nf
  !! internal
  integer :: i, ns
  ns = size(functs)
  if(ns<1) _die('ns<1')
  nf = 0
  do i=1,ns;  nf = max(nf, size(functs(i)%mu2j)); enddo
end function ! get_nmult_max


end module ! modul_funct_l_mult
