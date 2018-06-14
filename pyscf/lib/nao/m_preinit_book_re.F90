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

module m_preinit_book_re

#include "m_define_macro.F90" 
  use m_die, only : die

  implicit none
  private die
 
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
subroutine preinit_book_re(pb, p2b)
  use m_book_pb, only : book_pb_t
  use m_prod_basis_type, only : prod_basis_t, get_sv_ptr
  use m_system_vars, only : system_vars_t, get_natoms
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  type(book_pb_t), allocatable, intent(inout) :: p2b(:)
  !! internal
  type(system_vars_t), pointer :: sv
  integer :: natoms, i
  
  sv => get_sv_ptr(pb)
  natoms = get_natoms(sv)
  
  _dealloc(p2b)
  allocate(p2b(natoms))
  
  do i=1,natoms
    if(pb%book_dp(i)%top/=1) _die('!%top/=1')
    p2b(i) = pb%book_dp(i)
  enddo ! 
  
end subroutine !preinit_book_re   

end module !m_preinit_book_re
