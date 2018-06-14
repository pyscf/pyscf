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

module m_pb_hkernel_pack8
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_log, only : log_size_note, log_memory_note

  implicit none

contains


!
! Computes the Hartree Kernel in packed form
!
subroutine pb_hkernel_pack8(pb, cp, hkernel_pack, para,iv,ilog)
  use m_arrays, only : d_array2_t
  use m_parallel, only : para_t
  use m_prod_basis_type, only : prod_basis_t, get_book_type
  use m_book_pb, only : book_pb_t
  use m_coul_param, only : coul_param_t
  use m_pb_coul_aux, only : pb_coul_aux_t, init_aux_pb_cp, alloc_init_ls_blocks, dealloc
  use m_pb_coul_pack8, only : pb_coul_pack8
  ! external
  type(prod_basis_t), intent(in) :: pb
  type(coul_param_t), intent(in) :: cp
  real(8), intent(inout) :: hkernel_pack(:)
  integer, intent(in) :: iv, ilog
  type(para_t), intent(in) :: para
  
  ! internal
  type(book_pb_t), allocatable  :: ls_blocks(:,:)
  type(pb_coul_aux_t) :: ca
  
  if(iv>0)write(ilog,'(a,a)')'comp_hkernel_pack: enter... ', __FILE__
  if(iv>0)write(ilog,'(a,i6)')'comp_hkernel_pack: get_book_type(pb) ', get_book_type(pb)

  call init_aux_pb_cp(pb, cp, ca, iv)
  call alloc_init_ls_blocks(pb, ls_blocks, .false.)
  call pb_coul_pack8(ca, ls_blocks, hkernel_pack, para, iv)

  _dealloc(ls_blocks)
  call dealloc(ca)

end subroutine ! pb_hkernel_pack8



!
!
!
subroutine alloc_comp_kernel_pack_pb(pb, ctype, scr_const, use_mult, vC_pack, para, iv, ilog)
  use m_parallel, only : para_t
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_gen, only : get_nfunct_prod_basis
  use m_log, only : log_timing_note
  use m_coul_param, only : coul_param_t, init_coul_param_expl
  
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  character(*), intent(in) :: ctype
  real(8), intent(in) :: scr_const
  integer, intent(in) :: use_mult
  real(8), intent(inout), allocatable :: vC_pack(:)
  type(para_t), intent(in) :: para
  integer, intent(in) :: iv, ilog
  !! internal
  type(coul_param_t) :: cp
  real(8) :: t1, t2
  !! Dimensions
  integer :: nf_pb
  nf_pb = get_nfunct_prod_basis(pb)
  !! END of Dimensions
 
  call cputime(t1)
  _dealloc(vC_pack)
  allocate(vC_pack(nf_pb*(nf_pb+1)/2))
  call init_coul_param_expl(ctype, scr_const, use_mult, cp)
!  call init_aux_pb_cp(pb, cp, aux, iv)
  
  call pb_hkernel_pack8(pb, cp, vC_pack, para,iv,ilog)

  call cputime(t2)
  call log_timing_note('alloc_comp_kernel_pack_pb_total', t2-t1, iv)

end subroutine ! alloc_comp_kernel_pack_pb


end module !m_pb_hkernel_pack8
