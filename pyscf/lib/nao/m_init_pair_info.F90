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

module m_init_pair_info

#include "m_define_macro.F90" 
  use m_die, only : die

  implicit none
  private die
 
  !! The structure describes pairs of sets of radial orbitals 
  !! (pseudo-atomic orbitals in case of SIESTA). The main reason for this 
  !! is all-electron calculations with more than two shells for which we have 
  !! to generate the bilocal products for each pair of radial orbitals optimally
  !! choosing carefully the expansion center according to the spatial extension
  !! of the radial orbitals. The center must be closer to the short-ranged orbital.
  !! 
  !!   Example of construction of array of pair_info_t type elements is in the module
  !! m_domiprod, in the subroutine constr_list_bi_pairs_info(...)
!  type pair_info_t
!    integer :: atoms(2)=-999   ! it is allowed not to initialize the 'atoms' field
!    integer :: species(2)=-999
!    real(8) :: coords(3,2)=-999
!    integer :: cells(3,2)=-999
!    integer :: ls2nrf(2) = -999 ! ! a correspondence : local specie (1 or 2) --> number of radial functions
!    integer, allocatable :: rf_ls2mu(:,:) ! ! a correspondence : radial function, local specie (1 or 2) --> "multiplett" in system_vars_t
!  end type ! pair_info_t

  contains

!
! Initialization of the pair_info_t in a simplest case, when only specie-indices and coordinates are given.
!
subroutine init_pair_info(sp12, rc12, ncc, cc2atom, sv, info)
  use m_system_vars, only : system_vars_t
  use iso_c_binding, only: c_double, c_int64_t
  use m_pair_info, only : pair_info_t
  
  implicit none
  !! external
  integer, intent(in) :: sp12(:)
  real(c_double), intent(in) :: rc12(:,:)
  integer(c_int64_t), intent(in) :: ncc
  integer(c_int64_t), intent(in) :: cc2atom(:)
  type(system_vars_t), intent(in) :: sv
  type(pair_info_t), intent(inout) :: info
  !! internal
  integer :: ls, rf, nrfmx
  
  info%atoms = -1
  info%species(1:2) = int(sp12(1:2))
  info%coords(1:3,1:2) = rc12(1:3,1:2)
  info%cells = 0
  info%ls2nrf(1:2) = sv%uc%sp2nmult(sp12(1:2))
  nrfmx = maxval(info%ls2nrf)
  _dealloc(info%rf_ls2mu)
  allocate(info%rf_ls2mu(nrfmx,2))
  info%rf_ls2mu = -999
  do ls=1,2; do rf=1,info%ls2nrf(ls); info%rf_ls2mu(rf,ls) = rf; enddo; enddo 

  _dealloc(info%cc2atom)
  info%ncc = int(ncc)
  if(ncc>0) then
    if(any(cc2atom(1:ncc)<1)) _die('cc2atom(1:ncc)<1')
    allocate(info%cc2atom(ncc))
    info%cc2atom(:) = int(cc2atom(:))
  endif

end subroutine ! init_pair_info

end module !m_init_pair_info
