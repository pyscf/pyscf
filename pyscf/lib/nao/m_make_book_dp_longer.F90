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

module m_make_book_dp_longer

  use m_precision, only : blas_int
#include "m_define_macro.F90" 
  use m_die, only : die
  use m_warn, only : warn
  
  implicit none
  private die
  private warn
  !private get_cdatetime

  contains

!
! Copies first entries in book_dp and enlarges book_dp by number of bilocal pairs
!
subroutine make_book_dp_longer(nbp, pb)
  use m_prod_basis_type, only : prod_basis_t
  use m_system_vars, only : get_natoms
  use m_book_pb, only : book_pb_t
  
  implicit none
  integer, intent(in) :: nbp
  type(prod_basis_t), intent(inout) :: pb
  !Internal
  integer :: natoms
  type(book_pb_t), allocatable :: book_dp_local(:)

  natoms = get_natoms(pb%sv)
  allocate(book_dp_local(natoms))
  if(.not. allocated(pb%book_dp)) _die('!book_dp')
  if(size(pb%book_dp)<natoms) _die('size(pb%book_dp)<natoms')
  if(nbp<1) _die('!nbp<1')
  book_dp_local(1:natoms) = pb%book_dp(1:natoms)
  _dealloc( pb%book_dp )
  allocate(pb%book_dp(nbp+natoms))
  pb%book_dp(1:natoms) = book_dp_local(1:natoms)
  
end subroutine !make_book_dp_longer

end module !m_make_book_dp_longer
