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

module m_hkernel_pb_bcrs8

#include "m_define_macro.F90"

  implicit none
  
  contains

!
!
!
subroutine hkernel_pb_bcrs(pb, hxc)
  use m_hxc8, only : hxc8_t
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_gen, only : get_nfunct_prod_basis
  use m_coul_param, only : coul_param_t, init_coul_param_expl
  use m_pb_ls_blocks_ovrlp_all, only : pb_ls_blocks_ovrlp_all
  use m_pb_c2nf, only : pb_c2nf
  use m_pb_coul_bcrs8, only : pb_coul_bcrs
  use m_block_crs8, only : init
  use m_pb_coul_aux, only : init_aux_pb_cp
  
  implicit none
  ! external
  type(prod_basis_t), intent(in) :: pb
  type(hxc8_t), intent(inout) :: hxc
  ! internal
  integer :: nf_pb
  type(coul_param_t) :: cp
  integer, allocatable :: ic2size(:), blk2cc(:,:)

  nf_pb = get_nfunct_prod_basis(pb)
  call init_coul_param_expl("HARTREE", 0D0, 1, cp)
  call pb_c2nf(pb, ic2size)
  call pb_ls_blocks_ovrlp_all(pb, blk2cc)
  call init(ic2size, blk2cc, "N", hxc%bcrs)
  call init_aux_pb_cp(pb, cp, hxc%ca)
  call pb_coul_bcrs(hxc%ca, hxc%bcrs)

  _dealloc(ic2size)
  _dealloc(blk2cc)

end subroutine !hkernels_pb_bcrs8


end module !m_hkernel_pb_bcrs8
