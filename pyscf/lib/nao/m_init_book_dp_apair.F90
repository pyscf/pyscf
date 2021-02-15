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

module m_init_book_dp_apair

#include "m_define_macro.F90"
  use m_die, only : die
  
  implicit none
  private die

contains

!
! A custom initialization srtep for libint in the operation of atom-pair-by-atom-pair
!
subroutine init_book_dp_apair(pb)
  use m_prod_basis_type, only : prod_basis_t, set_book_type
  use m_system_vars, only : get_natoms, get_nr
  use m_pair_info, only : pair_info_t
  use m_uc_skeleton, only : get_sp, get_nmult, get_coord
  use m_sph_bes_trans, only : Talman_plan_t, sbt_plan, sbt_destroy
  use m_functs_l_mult_type, only : init_functs_lmult_mom_space, get_jcutoff_lmult
  use m_preinit_book_re, only : preinit_book_re  
  use m_preinit_book_dp, only : preinit_book_dp

  implicit none
  !! external
  type(prod_basis_t), intent(inout) :: pb
  !! internal
  integer :: natoms, atom, sp,nrf,ls,rf,jmx_lp,npairs
  type(pair_info_t), allocatable :: p2i(:)
  type(Talman_plan_t) :: Talman_plan
  natoms = get_natoms(pb%sv)
  
  allocate(p2i(natoms+1))
  do atom=1,natoms
    sp = get_sp(pb%sv%uc, atom)
    p2i(atom)%atoms(1:2) = [atom,atom]
    p2i(atom)%species(1:2) = [sp,sp]
    p2i(atom)%cells(:,:) = 0
    p2i(atom)%coords(1:3,1) = get_coord(pb%sv%uc, atom)
    p2i(atom)%coords(1:3,2) = p2i(atom)%coords(1:3,1)
    p2i(atom)%ls2nrf = get_nmult(pb%sv%uc, sp)
    nrf = p2i(atom)%ls2nrf(1)
    allocate(p2i(atom)%rf_ls2mu(nrf,2))
    do ls=1,2; do rf=1,nrf; p2i(atom)%rf_ls2mu(rf,ls)=rf; enddo; enddo ! rf;ls
  enddo ! atom
  p2i(natoms+1)%atoms(1:2) = 1  ! this is an imitation of bilocal pair
  p2i(natoms+1)%species(1:2) = [get_sp(pb%sv%uc, 1),get_sp(pb%sv%uc, 1)]  ! this is an imitation of bilocal pair
  p2i(natoms+1)%cells(1:3,1) = 1 ! this is an imitation of bilocal pair
  p2i(natoms+1)%cells(1:3,2) = 2 ! this is an imitation of bilocal pair
  p2i(natoms+1)%ls2nrf = -999  ! this is an imitation of bilocal pair

  call preinit_book_dp(pb, p2i, pb%book_dp)
  
  _dealloc(pb%sp_biloc2functs)
  allocate(pb%sp_biloc2functs(1))
  _dealloc(pb%sp_biloc2vertex)
  allocate(pb%sp_biloc2vertex(1))

  call preinit_book_re(pb, pb%book_re)  ! Because we are so determined, we can initialize the reexpressing bookkeping already now, i.e. after local products are generated.
  npairs = size(pb%book_dp)
  if(npairs<1) _die('!npairs')
  _dealloc(pb%coeffs)
  allocate(pb%coeffs(npairs))
  call set_book_type(2, pb) ! This know for certain already here because we are determined not to have any dominant products stored after generation of the basis
  jmx_lp = get_jcutoff_lmult(pb%sp_local2functs)
  call sbt_plan(Talman_plan, get_nr(pb%sv), jmx_lp, pb%rr, pb%pp, .true.)
  call init_functs_lmult_mom_space(Talman_plan, pb%sp_local2functs, pb%sp_local2functs_mom)
  call sbt_destroy(Talman_plan)
  
end subroutine !init_book_dp_apair 

end module !m_init_book_dp_apair
