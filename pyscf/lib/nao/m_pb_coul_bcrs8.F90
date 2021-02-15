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

module m_pb_coul_bcrs8
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_log, only : log_size_note, log_memory_note

  implicit none

contains

!
! Evaluates two-center coulomb integrals between products
!
subroutine pb_coul_bcrs(a, hk)
  use m_block_crs8, only : block_crs8_t, get_nblocks_stored
  use m_book_pb, only : book_pb_t
  use m_parallel, only : para_t
  use m_prod_basis_gen, only : get_book
  use m_prod_basis_type, only : get_nc_book, get_coord_center
  use m_prod_basis_type, only : get_rcut_per_center, get_nfunct_per_book
  use m_prod_basis_type, only : get_spp_sp_fp
  use m_pb_coul_aux, only : pb_coul_aux_t, alloc_runtime_coul, comp_aux
  use m_pb_coul_aux, only : comp_kernel_block_22, comp_kernel_block_21
  use m_pb_coul_11, only : pb_coul_11
  use m_pb_coul_12, only : comp_kernel_block_12
  
  implicit  none
  ! external
  type(pb_coul_aux_t), intent(in), target :: a
  type(block_crs8_t), intent(inout) :: hk

  ! internal
  type(book_pb_t) :: book1, book2
  integer :: spp1, spp2, ic1, ic2, size1, size2, cc(2)
  integer :: ncpb1,ncpb2,icpb1,icpb2, spp2a(3), spp1a(3), si, fi, sj, fj
  real(8) :: rcut1, rcut2, Rvec2(3), Rvec1(3)
  integer(8) :: block, sfn(3)
  real(8), allocatable :: tmp(:), bessel_pp(:,:), f1_mom(:), f1f2_mom(:)
  real(8), allocatable :: r_scalar_pow_jp1(:), real_overlap(:,:)

  !! multipoles or overlapping orbitals
  logical :: is_overlap, upper_only
  complex(8), allocatable :: ylm_thrpriv(:)
  real(8), allocatable :: S(:), wigner_matrices1(:,:,:), wigner_matrices2(:,:,:)
  real(8), allocatable :: array(:,:)
  !! Parallelization over nodes (MPI)
  integer :: type_of_mu1,type_of_mu2, nb

  nb = get_nblocks_stored(hk)
  
!! Loop over atomic quadruplets
!$OMP PARALLEL DEFAULT(NONE) &
!$OMP PRIVATE (block,ic1,ic2,bessel_pp,f1_mom,f1f2_mom,type_of_mu1,type_of_mu2)&
!$OMP PRIVATE (tmp, is_overlap, r_scalar_pow_jp1, real_overlap) &
!$OMP PRIVATE (ylm_thrpriv, S, rcut1, rcut2, Rvec1, Rvec2,book1,book2) &
!$OMP PRIVATE (wigner_matrices1,wigner_matrices2, upper_only) &
!$OMP PRIVATE (size1, size2,spp1, spp2,cc)&
!$OMP PRIVATE (ncpb1,ncpb2,icpb1,icpb2, spp2a, spp1a, array, si, fi, sj, fj, sfn)&
!$OMP SHARED (a,hk,nb)

  allocate(bessel_pp(a%nr, 0:2*a%jcutoff));
  allocate(f1_mom(a%nr));
  allocate(f1f2_mom(a%nr));
  allocate(real_overlap(-a%jcutoff:a%jcutoff,-a%jcutoff:a%jcutoff));
  allocate(ylm_thrpriv(0:(2*a%jcutoff+1)**2));
  allocate(S(0:2*a%jcutoff));
  allocate(r_scalar_pow_jp1(0:2*a%jcutoff));
  allocate(tmp(-a%jcutoff:a%jcutoff))
  allocate(wigner_matrices1(-a%jcutoff:a%jcutoff, -a%jcutoff:a%jcutoff, 0:a%jcutoff));
  allocate(wigner_matrices2(-a%jcutoff:a%jcutoff, -a%jcutoff:a%jcutoff, 0:a%jcutoff));

!!! HERE is the loop to be MPI parallelized
!$OMP DO SCHEDULE(DYNAMIC,1)
  do block=1,nb
    cc(1:2) = hk%blk2cc(1:2,block)
    book1 = get_book(a%pb, cc(1))
    book2 = get_book(a%pb, cc(2))
    
    ic1 = book1%ic
    ic2 = book2%ic

    type_of_mu1 = book1%top
    spp1    = book1%spp
    size1   = get_nfunct_per_book(a%pb, book1)
    
    type_of_mu2 = book2%top
    spp2    = book2%spp
    size2   = get_nfunct_per_book(a%pb, book2)

    ncpb1 = get_nc_book(a%pb, book1)
    ncpb2 = get_nc_book(a%pb, book2)

    _dealloc(array)
    allocate(array(size1,size2))

    do icpb2=1,ncpb2
      Rvec2 = get_coord_center(a%pb, book2, icpb2)
      rcut2 = get_rcut_per_center(a%pb, book2, icpb2)
      spp2a = get_spp_sp_fp(a%pb, book2, icpb2)
    do icpb1=1,ncpb1
      Rvec1 = get_coord_center(a%pb, book1, icpb1)
      rcut1 = get_rcut_per_center(a%pb, book1, icpb1)
      spp1a = get_spp_sp_fp(a%pb, book1, icpb1)


      call comp_aux(a, rcut1, Rvec1, rcut2, Rvec2, ylm_thrpriv, is_overlap, &
        r_scalar_pow_jp1, bessel_pp)        

    
      if(type_of_mu1==1 .and. type_of_mu2==1 ) then

        call pb_coul_11( a, spp1, spp2, is_overlap, array, f1f2_mom, S, &
          real_overlap, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)

      else if(type_of_mu1==1 .and. type_of_mu2==2) then

        wigner_matrices2 = a%pair2wigner_matrices(:,:,:,ic2)
      
        call comp_kernel_block_12( a, spp1, spp2a, is_overlap, wigner_matrices2, array, &
          f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)

      else if(type_of_mu1==2 .and. type_of_mu2==1) then

        wigner_matrices1 = a%pair2wigner_matrices(:,:,:,ic1)

        call comp_kernel_block_21( a, spp1a, spp2, is_overlap, wigner_matrices1, array, &
          f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)

      else if(type_of_mu1==2 .and. type_of_mu2==2) then
      
        wigner_matrices1 = a%pair2wigner_matrices(:,:,:,ic1) 
        wigner_matrices2 = a%pair2wigner_matrices(:,:,:,ic2)
        if(ic1==ic2) then; upper_only = .true.; else; upper_only = .false.; endif
      
        call comp_kernel_block_22(a, spp1a, spp2a, is_overlap, &
          wigner_matrices1, wigner_matrices2, array, &
          f1_mom, f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
        
      else
        write(0,*) 'type_of_mu1, type_of_mu2', type_of_mu1, type_of_mu2
        _die('unknown type_of_mu1 or type_of_mu2')
      endif
    
    enddo ! icpb1
    enddo ! icpb2
    
    si = book1%si(3)
    fi = book1%fi(3)

    sj = book2%si(3)
    fj = book2%fi(3)
    
    sfn = hk%blk2sfn(1:3,block)
    hk%d(sfn(1):sfn(2)) = reshape(array, [sfn(3)])
    
  end do ! block
  !$OMP END DO
  _dealloc(array)
  _dealloc(bessel_pp)
  _dealloc(f1_mom)
  _dealloc(f1f2_mom)
  _dealloc(real_overlap)
  _dealloc(ylm_thrpriv)
  _dealloc(S)
  _dealloc(r_scalar_pow_jp1)
  _dealloc(tmp)
  _dealloc(wigner_matrices1)
  _dealloc(wigner_matrices2)
  !$OMP END PARALLEL

end subroutine !pb_coul_bcrs

end module !m_pb_coul_pack8
