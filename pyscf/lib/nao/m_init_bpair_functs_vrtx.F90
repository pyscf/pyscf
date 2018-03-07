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

module m_init_bpair_functs_vrtx

#include "m_define_macro.F90" 
  use m_die, only : die

  implicit none
  private die
 

  contains

!
!
!
subroutine init_bpair_functs_vrtx(a, pair_info, m2nf, evals, ff2, &
  vertex_real2, lready, rcut, center, dp_a, fmm_mem, vmm)
  use m_prod_basis_param, only : prod_basis_param_t
  use m_prod_basis_param, only : get_eigmin_bilocal, get_jcutoff
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t
  use m_dp_aux, only : dp_aux_t
  use m_prod_basis_type, only : functs_m_mult_t, vertex_3cent_t
  use m_prod_basis_type, only : prod_basis_t
  use m_pair_info, only : pair_info_t, get_rf_ls2so
  use m_functs_m_mult_type, only : gather, dealloc
  use m_mmult_normalize, only : mmult_normalize
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: pair_info
  integer, allocatable, intent(in) :: m2nf(:)
  real(8), intent(in), allocatable :: evals(:,:,:)
  real(8), intent(in), allocatable :: ff2(:,:,:,:,:)
  real(8), intent(in), allocatable :: vertex_real2(:,:,:,:,:)
  logical, intent(in) :: lready 
  real(8), intent(in) :: rcut
  real(8), intent(in) :: center(:)
  type(dp_aux_t), intent(in) :: dp_a
  real(8), intent(inout), allocatable :: fmm_mem(:)
  !type(functs_m_mult_t), intent(inout) :: fmm
  type(vertex_3cent_t), intent(inout) :: vmm
!  type(prod_basis_t), intent(inout) :: pb
 
  !! internal
  real(8) :: bilocal_eigmin, pi
  integer :: npmax, m, n, np, k, nev, jcutoff, j12_mx, no(2)
  integer :: atoms(2), nn3(3), nppp, prod, j
  type(functs_m_mult_t) :: fmm
  
  pi = 4D0*atan(1d0)

  _dealloc(fmm%ir_j_prd2v)
  _dealloc(fmm%prd2m)
  _dealloc(vmm%vertex)

  !! Initial work: allocation/preinitialisation of vertex coefficients
  no(1:2)  = dp_a%sp2norbs(pair_info%species)
  if(lready) then
    npmax = -1
    fmm%nfunct = npmax   
    call gather(fmm, fmm_mem)
    call dealloc(fmm)
    return
  endif

  nev = size(evals,2)
  jcutoff = get_jcutoff(a%pb_p)
  atoms(1:2) = pair_info%atoms(1:2)

  _dealloc(fmm%crc)
  allocate(fmm%crc(7,1))
  fmm%crc = -999

  !! Determine the number of bilocal product functions !!!!
  j12_mx = ubound(m2nf,1)
  bilocal_eigmin = get_eigmin_bilocal(a%pb_p)  
  npmax = 0;
  nppp  = 0
  do m=-j12_mx, j12_mx
    n = m2nf(m);
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
    enddo ! k
    npmax = max(np, npmax)
    nppp  = nppp + np
  enddo ! m
  fmm%nfunct = nppp
  if (npmax < 1) then
    call gather(fmm, fmm_mem)
    call dealloc(fmm)
    return
  endif  
 
  nn3 = [a%nr,a%pb_p%jcutoff,nppp]
  allocate(fmm%ir_j_prd2v(nn3(1),0:nn3(2),nn3(3)))
  fmm%ir_j_prd2v = 0
  
  allocate(fmm%prd2m(nppp))
  fmm%prd2m = -999

  nn3 = [no(1:2),nppp]
  allocate(vmm%vertex(nn3(1),nn3(2),nn3(3)))
  vmm%vertex = 0
  !! END of Initial work: allocation/preinitialsation of vertex coefficients 
  
  fmm%crc(1:7,1) = [center(1:3), rcut, 1D0*1, 1D0*nppp, 1D0*nppp]

  fmm%coords  = pair_info%coords ! two centers because functionals are given in a rotated frame
  fmm%cells   = pair_info%cells  ! 
  fmm%species = pair_info%species! atom species
  fmm%atoms   = pair_info%atoms  ! atoms
  fmm%rcuts   = dp_a%sp2rcut(pair_info%species)! Spatial cutoff of the orbital functions
  fmm%nr = size(fmm%ir_j_prd2v,1)
  fmm%jmax = ubound(fmm%ir_j_prd2v,2)

  vmm%centers = -999
  
  prod = 0
  do m=-j12_mx,j12_mx
    n = m2nf(m)
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
      prod = prod + 1
      !! Products initialization (bilocal)
      fmm%prd2m(prod) = m
      do j=abs(m),a%pb_p%jcutoff
        fmm%ir_j_prd2v(1:a%nr,j,prod) = sqrt(4*pi/(2*j+1))*ff2(1:a%nr,j,k,m,2)
      enddo

      vmm%vertex(:,:,prod) = vertex_real2(m,k,1:no(1),1:no(2),2)
    enddo ! mprod
  enddo ! m

  if(a%pb_p%normalize_dp>0) call mmult_normalize(a%rr, fmm, vmm)

  call gather(fmm, fmm_mem)
  call dealloc(fmm)
  
end subroutine ! init_bpair_functs_vrtx


end module !m_init_bpair_functs_vrtx
