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

module m_apair_size

  use iso_c_binding, only: c_double, c_int64_t
#include "m_define_macro.F90"

  implicit none

  contains

!
!
!
subroutine apair_size(pb, pair, n)
  use m_system_vars, only : system_vars_t
  use m_prod_basis_type, only : prod_basis_t, get_vertex_dp_ptr, get_book_dp_ptr, get_coeffs_pp_ptr
  use m_book_pb, only : book_pb_t
  
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer(c_int64_t), intent(inout) :: n
  !! internal
  type(book_pb_t), pointer :: bk
  real(c_double), pointer :: vrtx(:,:,:), cc(:,:)
  
  bk => get_book_dp_ptr(pb, pair)
  if( bk%ic<1 ) then; write(6,*)__FILE__, __LINE__, bk%ic; stop '!%ic<1'; endif
  vrtx => get_vertex_dp_ptr(pb, pair)
  n = 99 + size(vrtx)
  if (allocated(pb%coeffs)) then
    cc   => get_coeffs_pp_ptr(pb, pair)
    n = n + size(cc) + 2*size(pb%coeffs(pair)%ind2book_re)
  endif

end subroutine ! apair_size

end module ! m_apair_size
