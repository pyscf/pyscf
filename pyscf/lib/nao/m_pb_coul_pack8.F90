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

module m_pb_coul_pack8
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_log, only : log_size_note, log_memory_note

  implicit none

contains

!
!
!
subroutine pb_coul_pack8(aux, ls_blocks, hk_pack, para, iv_in)
  use m_arrays, only : d_array2_t
  use m_log, only : log_size_note
  use m_parallel, only : para_t
  use m_book_pb, only : book_pb_t
  use m_pb_coul_aux, only : pb_coul_aux_t,distr_blocks 

  ! external
  type(pb_coul_aux_t), intent(in) :: aux
  type(book_pb_t), allocatable, intent(in) :: ls_blocks(:,:)
  real(8), intent(inout) :: hk_pack(:)
  type(para_t), intent(in) :: para
  integer, intent(in), optional :: iv_in
  !! internal
  integer(8), allocatable :: node2size(:), node2displ(:)
  integer, allocatable :: node2fb(:), node2lb(:) ! node2first_block, node2last_block
  real(8), allocatable :: block2rt(:)
  integer :: iv

  if(present(iv_in)) then; iv = iv_in; else; iv = 0; endif

  call distr_blocks(ls_blocks, node2size, node2displ, node2fb, node2lb, para)
  call comp_kernel_coulomb_overlap_pack8(aux, node2fb, node2lb, ls_blocks, block2rt, &
    hk_pack, para)

  _dealloc(node2size)
  _dealloc(node2displ)
  _dealloc(node2fb)
  _dealloc(node2lb)
  _dealloc(block2rt)
  
end subroutine ! pb_coul_blocks


!
! Evaluates two-center coulomb integrals between products
!
subroutine comp_kernel_coulomb_overlap_pack8(aux, & 
  node2fb, node2lb, ls_blocks, block2rt, hk_pack, para)

  use m_arrays, only : d_array2_t
  use m_abramowitz, only : spherical_bessel
  use m_book_pb, only : book_pb_t, get_R2mR1
  use m_csphar, only : csphar
  use m_parallel, only : para_t
  use m_prod_basis_type, only : get_nc_book, get_coord_center
  use m_prod_basis_type, only : get_rcut_per_center, get_nfunct_per_book
  use m_prod_basis_type, only : get_spp_sp_fp
  use m_pb_coul_aux, only : pb_coul_aux_t, alloc_runtime_coul, comp_aux
  use m_pb_coul_aux, only : comp_kernel_block_22, comp_kernel_block_21
  use m_pb_coul_12, only : comp_kernel_block_12
  use m_pb_coul_11, only : pb_coul_11
  use m_pack_matrix, only : put_block_pack_mat88
  
  implicit  none
  ! external
  type(pb_coul_aux_t), intent(in), target :: aux
  integer, intent(in), allocatable  :: node2fb(:), node2lb(:)
  type(book_pb_t), intent(in)  :: ls_blocks(:,:)
  real(8), allocatable, intent(inout)  :: block2rt(:)
  real(8), intent(inout) :: hk_pack(:)
  type(para_t), intent(in) :: para

  ! internal
  integer :: spp1, spp2, ic1, ic2, size1, size2, fb, lb
  integer :: ncpb1,ncpb2,icpb1,icpb2, spp2a(3), spp1a(3), si, fi, sj, fj
  real(8) :: r_scalar, rcut1, rcut2, scr_const, Rvec2(3), Rvec1(3)
  real(8), pointer :: pp(:), scr_pp(:)
  integer(8) :: block
  real(8), allocatable :: tmp(:), bessel_pp(:,:), f1_mom(:), f1f2_mom(:), array_aux(:,:)
  real(8), allocatable :: r_vec(:), r_scalar_pow_jp1(:), real_overlap(:,:)

  !! multipoles or overlapping orbitals
  logical :: is_overlap, upper_only
  complex(8), allocatable :: ylm_thrpriv(:)
  real(8), allocatable :: S(:), wigner_matrices1(:,:,:), wigner_matrices2(:,:,:)
  real(8), allocatable :: array(:,:)
  !! Parallelization over nodes (MPI)
  real(8) :: block_start_time, block_finish_time;
  integer :: type_of_mu1,type_of_mu2
  
#ifdef TIMING
  call alloc_runtime_coul(node2fb,node2lb,block2rt,para)
#endif

  fb = node2fb(para%rank)
  lb = node2lb(para%rank)

  pp=>aux%pp
  scr_pp => aux%scr_pp 
  scr_const = aux%scr_const
  
!! Loop over atomic quadruplets
!$OMP PARALLEL DEFAULT(NONE) &
!$OMP PRIVATE (block,ic1,ic2,bessel_pp,f1_mom,f1f2_mom,type_of_mu1,type_of_mu2)&
!$OMP PRIVATE (tmp, is_overlap, r_scalar_pow_jp1, real_overlap,r_scalar) &
!$OMP PRIVATE (ylm_thrpriv, S, rcut1, rcut2, array_aux, Rvec1, Rvec2) &
!$OMP PRIVATE (wigner_matrices1,wigner_matrices2, upper_only, r_vec) &
!$OMP PRIVATE (size1, size2, block_start_time, block_finish_time,spp1, spp2)&
!$OMP PRIVATE (ncpb1,ncpb2,icpb1,icpb2, spp2a, spp1a, array, si, fi, sj, fj)&
!$OMP SHARED (block2rt, ls_blocks,para, node2fb, node2lb,aux) &
!$OMP SHARED (pp, scr_pp, scr_const, hk_pack)

  allocate(bessel_pp(aux%nr, 0:2*aux%jcutoff));
  allocate(f1_mom(aux%nr));
  allocate(f1f2_mom(aux%nr));
  allocate(real_overlap(-aux%jcutoff:aux%jcutoff,-aux%jcutoff:aux%jcutoff));
  allocate(r_vec(3));
  allocate(ylm_thrpriv(0:(2*aux%jcutoff+1)**2));
  allocate(S(0:2*aux%jcutoff));
  allocate(r_scalar_pow_jp1(0:2*aux%jcutoff));
  allocate(tmp(-aux%jcutoff:aux%jcutoff))
  allocate(wigner_matrices1(-aux%jcutoff:aux%jcutoff, -aux%jcutoff:aux%jcutoff, 0:aux%jcutoff));
  allocate(wigner_matrices2(-aux%jcutoff:aux%jcutoff, -aux%jcutoff:aux%jcutoff, 0:aux%jcutoff));
  allocate(array_aux(aux%nfmx, aux%nfmx))

!!! HERE is the loop to be MPI parallelized
!$OMP DO SCHEDULE(DYNAMIC,1)
  do block = node2fb(para%rank), node2lb(para%rank)
#ifdef TIMING
    call cputime(block_start_time);
#endif

    ic1 = ls_blocks(1,block)%ic
    ic2 = ls_blocks(2,block)%ic

    type_of_mu1 = ls_blocks(1,block)%top
    spp1    = ls_blocks(1,block)%spp
    size1   = get_nfunct_per_book(aux%pb, ls_blocks(1,block))
    
    type_of_mu2 = ls_blocks(2,block)%top
    spp2    = ls_blocks(2,block)%spp
    size2   = get_nfunct_per_book(aux%pb, ls_blocks(2,block))

    ncpb1 = get_nc_book(aux%pb, ls_blocks(1,block))
    ncpb2 = get_nc_book(aux%pb, ls_blocks(2,block))

    _dealloc(array)
    allocate(array(size1,size2))

    do icpb2=1,ncpb2
      Rvec2 = get_coord_center(aux%pb, ls_blocks(2,block), icpb2)
      rcut2 = get_rcut_per_center(aux%pb, ls_blocks(2,block), icpb2)
      spp2a = get_spp_sp_fp(aux%pb, ls_blocks(2,block), icpb2)
    do icpb1=1,ncpb1
      Rvec1 = get_coord_center(aux%pb, ls_blocks(1,block), icpb1)
      rcut1 = get_rcut_per_center(aux%pb, ls_blocks(1,block), icpb1)
      spp1a = get_spp_sp_fp(aux%pb, ls_blocks(1,block), icpb1)
        
      call comp_aux(aux, rcut1, Rvec1, rcut2, Rvec2, ylm_thrpriv, is_overlap, &
        r_scalar_pow_jp1, bessel_pp)
        
!      r_vec = Rvec2 - Rvec1

!      call csphar(r_vec, ylm_thrpriv, 2*aux%jcutoff);

!      r_scalar = sqrt(sum(r_vec*r_vec));
!      if(aux%use_mult>0) then
!        is_overlap = (r_scalar<(rcut1+rcut2))
!      else
!        is_overlap = .true.
!      endif  

    
!      if(aux%logical_overlap) then
!        if(is_overlap) then ! only if the orbitals overlap we need this bessel function
!          do L=0,2*aux%jcutoff
!            do ir=1,aux%nr;
!              bessel_pp(ir,L)=pp(ir)**3 * spherical_bessel(L,r_scalar*pp(ir))
!            enddo !ir
!          enddo ! L
!        endif
!      else
!        if(is_overlap) then ! only if the orbitals overlap we need this bessel function
!          do L=0,2*aux%jcutoff
!            do ir=1,aux%nr;
!              bessel_pp(ir,L)=spherical_bessel(L,r_scalar*pp(ir))*pp(ir)
!            enddo !ir
!          enddo ! L

!          if(scr_const>0) then
!            do l=0,2*aux%jcutoff; bessel_pp(:,l)=bessel_pp(:,l)*scr_pp(:); enddo ! L
!          endif
               
!        else  !! non overlaping functions
!          do j=0,2*aux%jcutoff; r_scalar_pow_jp1(j) = 1.0D0/(r_scalar**(j+1)); enddo
!        endif !! is_overlap
!      endif  !! logical_sigma_overlap

    
    if(type_of_mu1==1 .and. type_of_mu2==1 ) then

      call pb_coul_11( aux, spp1, spp2, &
        is_overlap, array, f1f2_mom, S, real_overlap, &
        bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)

    else if(type_of_mu1==1 .and. type_of_mu2==2) then

      wigner_matrices2 = aux%pair2wigner_matrices(:,:,:,ic2)
      
      call comp_kernel_block_12( aux, spp1, spp2a, &
        is_overlap, wigner_matrices2, array, &
        f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)

    else if(type_of_mu1==2 .and. type_of_mu2==1) then

      wigner_matrices1 = aux%pair2wigner_matrices(:,:,:,ic1)

      call comp_kernel_block_21( aux, spp1a, spp2, is_overlap, &
        wigner_matrices1, array, f1f2_mom, S, real_overlap, &
        tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)

!      call csphar(-r_vec, ylm_thrpriv, 2*aux%jcutoff);
!      call comp_kernel_block_21_tr(aux, spp1, spp2, is_overlap, &
!        wigner_matrices1, block2result(block)%array, f1f2_mom, S, real_overlap, &
!        tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv, array_aux)

    else if(type_of_mu1==2 .and. type_of_mu2==2) then
      
      wigner_matrices1 = aux%pair2wigner_matrices(:,:,:,ic1) 
      wigner_matrices2 = aux%pair2wigner_matrices(:,:,:,ic2)
      if(ic1==ic2) then; upper_only = .true.; else; upper_only = .false.; endif
      
      call comp_kernel_block_22(aux, spp1a, spp2a, is_overlap, &
        wigner_matrices1, wigner_matrices2, array, &
        f1_mom, f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv);
        
    else
      write(0,*) 'type_of_mu1, type_of_mu2', type_of_mu1, type_of_mu2
      _die('unknown type_of_mu1 or type_of_mu2')
    endif
    
    enddo ! icpb1
    enddo ! icpb2
    
    si = ls_blocks(1,block)%si(3)
    fi = ls_blocks(1,block)%fi(3)

    sj = ls_blocks(2,block)%si(3)
    fj = ls_blocks(2,block)%fi(3)
    
    call put_block_pack_mat88(array, si,fi, sj,fj, hk_pack)
#ifdef TIMING
   call cputime(block_finish_time);
   block2rt(block)=block_finish_time-block_start_time;
#endif
  end do ! block
  !_G
  !$OMP END DO
  _dealloc(array)
  _dealloc(bessel_pp)
  deallocate(f1_mom)
  deallocate(f1f2_mom)
  deallocate(real_overlap)
  deallocate(r_vec)
  deallocate(ylm_thrpriv)
  deallocate(S)
  deallocate(r_scalar_pow_jp1)
  deallocate(tmp)
  deallocate(wigner_matrices1)
  deallocate(wigner_matrices2)
  deallocate(array_aux)
  !$OMP END PARALLEL

end subroutine !comp_kernel_coulomb_overlap8


end module !m_pb_coul_pack8
