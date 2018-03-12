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

module m_tci_ac_dp

#include "m_define_macro.F90"
  use m_die, only : die
  use m_timing, only : get_cdatetime
  implicit none

contains


!
! Evaluates two-center coulomb integrals between products
!
subroutine tci_ac_dp(aux, i2book, fmm_mem, array, &
  bessel_pp, f1f2_mom, roverlap, ylm_thrpriv, Sarr, r_scalar_pow_jp1, tmp)

  use m_pb_coul_aux, only : pb_coul_aux_t
  use m_book_pb, only : book_pb_t
  use m_prod_basis_type, only : get_i2s
  use m_functs_m_mult_type, only : functs_m_mult_t, get_nfunct_mmult, get_diff_sp
  use m_functs_m_mult_type, only : multipole_moms, funct_sbt, dealloc, scatter
  use m_wigner_rotation, only : simplified_wigner
    
  implicit  none
  ! external
  type(pb_coul_aux_t), intent(in), target :: aux
  type(book_pb_t), intent(in)  :: i2book(:)
!  type(functs_m_mult_t), intent(in) :: frea2_in
  real(8), intent(inout) :: fmm_mem(:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: tmp(:), bessel_pp(:,:), f1f2_mom(:)
  real(8), intent(inout), allocatable :: r_scalar_pow_jp1(:), roverlap(:,:)
  complex(8), intent(inout), allocatable :: ylm_thrpriv(:)
  real(8), intent(inout), allocatable :: Sarr(:)
  !! internal
  type(functs_m_mult_t) :: fmom2, mmom2, frea2
  integer :: si,fi,i,ni,jc,j,nf2
  integer, allocatable :: i2s(:)
  real(8), allocatable :: wmi(:,:), wm2(:,:,:)
  real(8) :: dR(3)!, diff
  
  
  call scatter(fmm_mem, frea2)
!  write(6,*) allocated(frea2%ir_j_prd2v), allocated(frea2_in%ir_j_prd2v), &
!    size(frea2%ir_j_prd2v), size(frea2_in%ir_j_prd2v)
!  diff = get_diff_sp(frea2_in, frea2)
!  write(6,*) __FILE__, __LINE__, diff
!  frea2 = frea2_in
  
  call get_i2s(aux%pb, i2book, i2s)
  ni = size(i2book)

  jc = aux%jcutoff
  allocate(wmi(-jc:jc,-jc:jc))
  allocate(wm2(-jc:jc,-jc:jc,0:jc))
  dR = frea2%coords(1:3,2)-frea2%coords(1:3,1)
  if(all(dR==0))dR=(/0D0, 0D0, 1D0/) !! this could happen with separate_core 1 option...
  do j=0,jc
    call simplified_wigner( dR, j, wmi(-j:j,-j:j), wm2(-j:j,-j:j,j))
  enddo

  nf2 = get_nfunct_mmult(frea2)
  call funct_sbt(aux%tp, frea2, fmom2)
  call multipole_moms(frea2, aux%pb%rr, mmom2)
  
  do i=1,ni
    si = i2s(i)
    fi = i2s(i+1)-1

    if(fi>size(array,1)) then
      write(6,*) si, fi, ubound(array), i
    endif

    if(nf2>size(array,2)) then
      write(6,*) si, fi, nf2, ubound(array), i
    endif
       
    call tci_ac_dp_inner(aux, i2book(i), wm2, fmom2, mmom2, array(si:fi,1:nf2), & 
      bessel_pp, f1f2_mom, roverlap, ylm_thrpriv, Sarr, r_scalar_pow_jp1, tmp)
         
  enddo

  _dealloc(i2s)
  _dealloc(wmi)
  _dealloc(wm2)
  call dealloc(fmom2)
  call dealloc(mmom2)
  call dealloc(frea2)
    
end subroutine !tci_ac_dp_loop

!
! Evaluates two-center coulomb integrals between products
! of type 1 (local, or l-multiplett) and type 2 (bilocal, or m-multiplett)
!
subroutine tci_ac_dp_inner(aux, b1, wm2, fmom2, mmom2, array, & 
  bessel_pp, f1f2_mom, real_overlap, &
  ylm_thrpriv, S, r_scalar_pow_jp1, tmp)

  use m_book_pb, only : book_pb_t
  use m_prod_basis_type, only : get_coord_center, get_spp_sp_fp
  use m_prod_basis_type, only : get_rcut_per_center, get_nfunct_per_book
  use m_pb_coul_aux, only : pb_coul_aux_t, comp_aux
  use m_pb_coul_12, only : pb_coul_12
  use m_functs_m_mult_type, only : functs_m_mult_t, get_nfunct_mmult
    
  implicit  none
  ! external
  type(pb_coul_aux_t), intent(in), target :: aux
  type(book_pb_t), intent(in)  :: b1
  real(8), intent(in)               :: wm2(:,:,:)
  type(functs_m_mult_t), intent(in) :: fmom2
  type(functs_m_mult_t), intent(in) :: mmom2
  real(8), intent(inout)            :: array(:,:)
  real(8), intent(inout), allocatable :: tmp(:), bessel_pp(:,:), f1f2_mom(:)
  real(8), intent(inout), allocatable :: r_scalar_pow_jp1(:), real_overlap(:,:)
  complex(8), intent(inout), allocatable :: ylm_thrpriv(:)
  real(8), intent(inout), allocatable :: S(:)

  ! internal
  integer :: spp1, ic1, size1, size2
  integer :: ncpb2,icpb2, spp2a(3), spp1a(3)
  real(8) :: rcut1, rcut2, Rvec2(3), Rvec1(3)

  logical :: is_overlap

  ic1 = b1%ic
  if(b1%top/=1) _die('!b1%top/=1')
  spp1  = b1%spp
  size1 = get_nfunct_per_book(aux%pb, b1)
  size2 = get_nfunct_mmult(fmom2)
  ncpb2 = size(fmom2%crc,2)

  array = 0

  do icpb2=1,ncpb2
    Rvec2 = fmom2%crc(1:3,icpb2)
    rcut2 = fmom2%crc(4,icpb2)
    spp2a = [-999, int(fmom2%crc(5:6,icpb2))]
    
    Rvec1 = get_coord_center(aux%pb, b1, 1)
    rcut1 = get_rcut_per_center(aux%pb, b1, 1)
    spp1a = get_spp_sp_fp(aux%pb, b1, 1)

    call comp_aux(aux, rcut1, Rvec1, rcut2, Rvec2, ylm_thrpriv, is_overlap, &
      r_scalar_pow_jp1, bessel_pp)
        
     
    call pb_coul_12( aux, spp1, spp2a, is_overlap, wm2, fmom2, mmom2, array, &
      f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)         

  enddo ! icpb2    
    
end subroutine !tci_ac_dp


end module !m_tci_ac_dp
