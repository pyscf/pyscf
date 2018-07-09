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

module m_pair_info

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
  type pair_info_t
    integer :: atoms(2)=-999   ! it is allowed not to initialize the 'atoms' field
    integer :: species(2)=-999
    real(8) :: coords(3,2)=-999
    integer :: cells(3,2)=-999
    integer :: ls2nrf(2) = -999 ! ! a correspondence : local specie (1 or 2) --> number of radial functions
    integer, allocatable :: rf_ls2mu(:,:) ! ! a correspondence : radial function, local specie (1 or 2) --> "multiplett" in system_vars_t
    integer :: ncc = -999 ! number of contributing centers
    integer, allocatable :: cc2atom(:) ! contributing center --> atom correspondence
  end type ! pair_info_t

  contains

!
!
!
subroutine dealloc_bp2info(bp2info)

  implicit none
  type(pair_info_t), intent(inout) :: bp2info

  _dealloc(bp2info%rf_ls2mu)
  _dealloc(bp2info%cc2atom)

end subroutine ! dealloc_bp2info

!
!
!
subroutine cp_pair_info(pi1, pi2)
  implicit none
  !! external
  type(pair_info_t), intent(in) :: pi1
  type(pair_info_t), intent(inout) :: pi2
  !! 
  integer :: nn(2)
  pi2%atoms=pi1%atoms
  pi2%species=pi1%species
  pi2%coords=pi1%coords
  pi2%cells=pi1%cells
  pi2%ls2nrf=pi1%ls2nrf
  
  if(allocated(pi1%rf_ls2mu)) then
    nn = ubound(pi1%rf_ls2mu)
_dealloc_u(pi2%rf_ls2mu, nn)
    if(.not. allocated(pi2%rf_ls2mu)) allocate(pi2%rf_ls2mu(nn(1),nn(2)))
    pi2%rf_ls2mu = pi1%rf_ls2mu
  else
    _dealloc(pi2%rf_ls2mu)
  endif   
  
end subroutine ! cp_pair_info  

!
!
!
subroutine get_rf_ls2so(ls2sp, ls2nrf, rf_ls2mu, mu_sp2j, rf_ls2so)
  implicit none
  integer, intent(in) :: ls2sp(2), ls2nrf(2), rf_ls2mu(:,:)
  integer, intent(in) :: mu_sp2j(:,:)
  integer, intent(inout) :: rf_ls2so(:,:)
  !! internal
  integer :: ls, fo, so, j, rf, mu

  !! figure out rf_ls2so, i.e. correspondence between radial function number, local specie --> start orbital
  rf_ls2so(:,:) = -999
  do ls=1,2
    fo=0
    do rf=1,ls2nrf(ls)
      mu = rf_ls2mu(rf,ls)
      j = mu_sp2j(mu, ls2sp(ls))
      so = fo+1; fo=so+2*j
      rf_ls2so(rf,ls) = so
    enddo ! rf
  enddo ! ls
  !! END of !! figure out rf_ls2so, i.e. correspondence between radial function number, local specie --> start orbital

end subroutine ! get_rf_ls2so

!
!
!
logical function exists_bilocal(pair2info)
  !! external
  type(pair_info_t), intent(in) :: pair2info(:)
  
  integer :: pair, npairs
  npairs = size(pair2info)
  exists_bilocal = .false.
  do pair=1,npairs
    exists_bilocal = is_bilocal(pair2info(pair))
    if(exists_bilocal) exit
  enddo

end function ! exists_bilocal

!
!
!
logical function exists_local(pair2info)
  !! external
  type(pair_info_t), intent(in) :: pair2info(:)
  
  integer :: pair, npairs
  npairs = size(pair2info)
  exists_local = .false.
  do pair=1,npairs
    exists_local = is_local(pair2info(pair))
    if(exists_local) exit
  enddo
end function ! exists_local

!
!
!
subroutine get_bilocal_pairs_only(pair2info, nbp, bp2info, iv)
  implicit none
  !! external
  type(pair_info_t), intent(in), allocatable :: pair2info(:) ! list of all pairs
  integer, intent(inout) :: nbp ! number of bilocal pairs
  type(pair_info_t), intent(inout), allocatable :: bp2info(:) ! list of bilocal pairs only
  integer, intent(in) :: iv
  !! internal
  integer :: step, pair
  integer :: npairs
  npairs = size(pair2info)
  
  do step=1,2
    nbp = 0
    do pair=1,npairs
      if(is_bilocal(pair2info(pair))) then
        nbp=nbp+1
        if(step==1) cycle
        call cp_pair_info(pair2info(pair), bp2info(nbp))
      endif  
    enddo ! pair
    if(step==2) cycle
    allocate(bp2info(nbp))
  enddo ! step
  
end subroutine ! get_bilocal_pairs_only  

!!
!!
!!
subroutine distr_atom_pairs_info(bilocal_pair2info, numnodes, &
  node2bilocal_natom_pairs, bilocal_pair_node2info)

  implicit none
  !! external
  type(pair_info_t), intent(in) :: bilocal_pair2info(:)
  integer, intent(in) :: numnodes
  integer, allocatable, intent(inout) :: node2bilocal_natom_pairs(:)
  type(pair_info_t), allocatable, intent(inout) :: bilocal_pair_node2info(:,:)

  !! internal
  integer :: node, max_pairs_per_node, pair_within_node, pair
  !! Dimensions
  integer :: bilocal_natom_pairs
  bilocal_natom_pairs = size(bilocal_pair2info)
  !! END of Dimensions

  allocate(node2bilocal_natom_pairs(0:numnodes-1))

  node = 0
  node2bilocal_natom_pairs = 0
  do pair=1, bilocal_natom_pairs
    node2bilocal_natom_pairs(node) = node2bilocal_natom_pairs(node) + 1
    node = node + 1
    if(node>numnodes-1) node = 0
  end do
  max_pairs_per_node = maxval(node2bilocal_natom_pairs)

  allocate(bilocal_pair_node2info(max_pairs_per_node, 0:numnodes-1));

  node = 0
  !bilocal_pair_node2info = 0
  node2bilocal_natom_pairs = 0
  do pair=1, bilocal_natom_pairs
    node2bilocal_natom_pairs(node) = node2bilocal_natom_pairs(node) + 1
    pair_within_node = node2bilocal_natom_pairs(node);
    bilocal_pair_node2info(pair_within_node, node) = bilocal_pair2info(pair)
    node = node + 1
    if(node>numnodes-1) node = 0
  end do

end subroutine !distr_atom_pairs_info

!
! Checks whether a pair is local or not
!
logical function is_local(i)
  implicit none
  type(pair_info_t), intent(in) ::  i  

  is_local = (i%atoms(1)==i%atoms(2)) &
    .and. (sum(abs(i%coords(:,1)-i%coords(:,2)))<1d-12)

end function !  is_local   

!
!
!
function is_bilocal(i)
  type(pair_info_t), intent(in) :: i
  logical :: is_bilocal
  
  is_bilocal = .not. is_local(i)
  
end function ! is_bilocal


end module !m_pair_info
