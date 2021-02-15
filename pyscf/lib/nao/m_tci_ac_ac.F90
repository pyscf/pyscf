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

module m_tci_ac_ac

#include "m_define_macro.F90"
  use m_die, only : die
  use m_timing, only : get_cdatetime
  implicit none

contains


!
! Evaluates two-center coulomb integrals between products
!
subroutine tci_ac_ac(aux, i2book, j2book, array, &
  bessel_pp, f1f2_mom, roverlap, ylm, Sarr, r_scalar_pow_jp1)

  use m_pb_coul_aux, only : pb_coul_aux_t, comp_aux
  use m_book_pb, only : book_pb_t
  use m_prod_basis_type, only : get_i2s
  use m_prod_basis_type, only : get_coord_center
  use m_prod_basis_type, only : get_rcut_per_center
  use m_pb_coul_11, only : pb_coul_11
  
  implicit  none
  ! external
  type(pb_coul_aux_t), intent(in), target :: aux
  type(book_pb_t), intent(in)  :: i2book(:), j2book(:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: bessel_pp(:,:), f1f2_mom(:)
  real(8), intent(inout), allocatable :: r_scalar_pow_jp1(:), roverlap(:,:)
  complex(8), intent(inout), allocatable :: ylm(:)
  real(8), intent(inout), allocatable :: Sarr(:)
  !! internal
  logical :: is_overlap
  integer :: si,fi,i,ni,sj,fj,j,nj,spp1,spp2
  integer, allocatable :: i2s(:), j2s(:)
  real(8) :: Rvec1(3), Rvec2(3), rcut1, rcut2
  
  if(any(i2book(:)%top/=1)) _die('!i2book(:)%top/=1')
  if(any(j2book(:)%top/=1)) _die('!j2book(:)%top/=1')
  
  call get_i2s(aux%pb, i2book, i2s)
  ni = size(i2book)

  call get_i2s(aux%pb, j2book, j2s)
  nj = size(j2book)

  do j=1,nj
    sj = j2s(j)
    fj = j2s(j+1)-1
    spp2 = j2book(j)%spp
    Rvec2 = get_coord_center(aux%pb, i2book(j), 1)
    rcut2 = get_rcut_per_center(aux%pb, i2book(j), 1)
    
    do i=1,ni
      si = i2s(i)
      fi = i2s(i+1)-1
      spp1 = i2book(i)%spp
      Rvec1 = get_coord_center(aux%pb, i2book(i), 1)
      rcut1 = get_rcut_per_center(aux%pb, i2book(i), 1)

      call comp_aux(aux, rcut1, Rvec1, rcut2, Rvec2, ylm, is_overlap, &
        r_scalar_pow_jp1, bessel_pp)
     
      call pb_coul_11(aux, spp1, spp2, is_overlap, array(si:fi,sj:fj), &
        f1f2_mom, Sarr, roverlap, bessel_pp, r_scalar_pow_jp1, ylm)
                
    enddo ! i
  enddo ! j

  _dealloc(i2s)
  _dealloc(j2s)
    
end subroutine !tci_ac_ac

!
! Evaluates two-center coulomb integrals between products
!
subroutine tci_ac_ac_cpy(aux, i2book, j2book, array, &
  bessel_pp, f1f2_mom, roverlap, ylm, Sarr, r_scalar_pow_jp1)

  use m_pb_coul_aux, only : pb_coul_aux_t, comp_aux
  use m_book_pb, only : book_pb_t
  use m_prod_basis_type, only : get_i2s
  use m_prod_basis_type, only : get_coord_center
  use m_prod_basis_type, only : get_rcut_per_center
  use m_pb_coul_11, only : pb_coul_11
  
  implicit  none
  ! external
  type(pb_coul_aux_t), intent(in), target :: aux
  type(book_pb_t), intent(in)  :: i2book(:), j2book(:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: bessel_pp(:,:), f1f2_mom(:)
  real(8), intent(inout), allocatable :: r_scalar_pow_jp1(:), roverlap(:,:)
  complex(8), intent(inout), allocatable :: ylm(:)
  real(8), intent(inout), allocatable :: Sarr(:)
  !! internal
  logical :: is_overlap
  integer :: si,fi,i,ni,sj,fj,j,nj,spp1,spp2
  integer, allocatable :: i2s(:), j2s(:)
!  real(8) :: Rvec1(3), Rvec2(3), rcut1, rcut2
  
  if(any(i2book(:)%top/=1)) _die('!i2book(:)%top/=1')
  if(any(j2book(:)%top/=1)) _die('!j2book(:)%top/=1')
  
  call get_i2s(aux%pb, i2book, i2s)
  ni = size(i2book)

  call get_i2s(aux%pb, j2book, j2s)
  nj = size(j2book)

  do j=1,nj
    sj = j2s(j)
    fj = j2s(j+1)-1
    spp2 = j2book(j)%spp
    
    do i=1,ni
      si = i2s(i)
      fi = i2s(i+1)-1
      spp1 = i2book(i)%spp
               
    enddo ! i
  enddo ! j

  _dealloc(i2s)
  _dealloc(j2s)
    
end subroutine !tci_ac_ac_cpy


end module !m_tci_ac_ac
