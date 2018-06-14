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

module m_preinit_book_dp

#include "m_define_macro.F90" 
  use m_die, only : die

  implicit none
  private die
 
!  type pair_info_t
!    integer :: atoms(2)=-999   ! it is allowed not to initialize the 'atoms' field
!    integer :: species(2)=-999
!    real(8) :: coords(3,2)=-999
!    integer :: cells(3,2)=-999
!    integer :: ls2nrf(2) = -999 ! ! a correspondence : local specie (1 or 2) --> number of radial functions
!    integer, allocatable :: rf_ls2mu(:,:) ! ! a correspondence : radial function, local specie (1 or 2) --> "multiplett" in system_vars_t
!  end type ! pair_info_t

!  !! Center to coordinate, shifts and specie pointer
!  type book_pb_t
!    integer :: ic=-999  ! number of the center in a global counting (A GLOBAL COUNTING is not necessarily THE GLOBAL COUNTING you would need...)
!    integer :: toc=-999 ! type of center: 1 -- local, 2 -- bilocal products
!    integer :: spp=-999 ! pointer to a product specie in a list of product species
!    real(8) :: coord(3)=-999   ! real space coordinate of the center without translations
!    integer :: cells(3,3)=-999 ! cells(1..3) number of unit cells along 1..3 direction on which the centers are shifted relative to 'first' unit cell
!    integer :: atoms(2)=-999   ! atoms (in the unit cell) for which the pair is created
!    integer :: si(3)=-999 !  start indices of the part in the global counting of real space degrees of freedom for atoms and prod basis
!    integer :: fi(3)=-999 ! finish indices of the part in the global counting of real space degrees of freedom for atoms and prod basis
!  end type ! book_pb_t

  contains

!
!
!
subroutine preinit_book_dp(pb, p2i, p2b)
  use m_book_pb, only : book_pb_t
  use m_pair_info, only : pair_info_t
  use m_prod_basis_type, only : prod_basis_t, get_sv_ptr
  use m_system_vars, only : get_atom2sfo_ptr, get_atom2sp_ptr
  use m_system_vars, only : system_vars_t, get_atom2coord_ptr
  use m_functs_l_mult_type, only : get_nfunct_lmult
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  type(pair_info_t), allocatable, intent(in)  :: p2i(:)
  type(book_pb_t), allocatable, intent(inout) :: p2b(:)
  !! internal
  type(system_vars_t), pointer :: sv
  integer :: npairs, i, cells(3,2), atoms(2), ibiloc_pair, ilocal_pair
  integer :: s3,f3,n3,nlp, spp
  integer, pointer :: atom2sp(:), atom2sfo(:,:)
  real(8), pointer :: atom2coord(:,:)
  
  if(.not. allocated(p2i)) _die('!p2i')
  npairs = size(p2i)
  
  sv => get_sv_ptr(pb)
  atom2sp => get_atom2sp_ptr(sv)
  atom2sfo => get_atom2sfo_ptr(sv)
  atom2coord => get_atom2coord_ptr(sv)
  
  _dealloc(p2b)
  allocate(p2b(npairs))
  
  ibiloc_pair = 0
  ilocal_pair = 0

  f3 = 0
  if(.not. allocated(pb%sp_local2functs)) _die('!sp_local2functs')
  nlp = size(pb%sp_local2functs)

  do i=1,npairs
    cells = p2i(i)%cells
    atoms = p2i(i)%atoms

    if(all(cells(:,1)==cells(:,2)) .and. atoms(1)==atoms(2)) then
      ilocal_pair=ilocal_pair + 1
      if(i/=ilocal_pair) _die('a "first local" convention is broken?')

      p2b(i)%top = 1 ! (i.e. local pair)
      spp = atom2sp(atoms(1))
      p2b(i)%spp  = spp
      p2b(i)%coord = atom2coord(1:3,atoms(1))
      
      if(spp<1 .or. spp>nlp) _die('!spp')
      s3 = f3 + 1
      n3 = get_nfunct_lmult(pb%sp_local2functs(spp))
      f3 = s3 + n3 - 1
      p2b(i)%si(3) = s3
      p2b(i)%fi(3) = f3
    else
      p2b(i)%top = 2 ! (i.e. bilocal pair)
      ibiloc_pair = ibiloc_pair + 1
      p2b(i)%spp  = ibiloc_pair
      p2b(i)%coord = -999 ! at this point I cannot guess --- it will be done during construction of pair
      p2b(i)%si(3) = -999 ! at this point I cannot guess --- it will be done during construction of pair
      p2b(i)%fi(3) = -999 ! at this point I cannot guess --- it will be done during construction of pair
    endif
    
    p2b(i)%ic = i
    p2b(i)%cells(1:3,1:2) = p2i(i)%cells(1:3,1:2)
    p2b(i)%cells(1:3,3)   = 0 ! this must be zero because dominant products always refer to unit cell.
    p2b(i)%atoms(1:2) = atoms(1:2)
    p2b(i)%si(1:2)    = atom2sfo(1,atoms)
    p2b(i)%fi(1:2)    = atom2sfo(2,atoms)
    
  enddo ! 
  
  
  
end subroutine !preinit_book_dp

end module !m_preinit_book_dp
