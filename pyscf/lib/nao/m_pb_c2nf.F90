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

module m_pb_c2nf

#include "m_define_macro.F90"
  implicit none

  contains

!
! Number of functions per center correspondence
!
subroutine pb_c2nf(pb, c2nf)
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_gen, only : get_nfunct_pbook, get_nbook
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(inout), allocatable :: c2nf(:)
  integer :: nc, ic
  
  nc = get_nbook(pb)
  _dealloc(c2nf)
  allocate(c2nf(nc))
  do ic=1,nc; c2nf(ic) = get_nfunct_pbook(pb, ic); enddo
  
end subroutine ! pb_c2nf 


end module ! m_pb_c2nf
