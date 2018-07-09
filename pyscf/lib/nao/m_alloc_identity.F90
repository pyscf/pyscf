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

module m_alloc_identity

#include "m_define_macro.F90"

  contains

!
!
!
subroutine alloc_identity(n, mat)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout), allocatable :: mat(:,:)
  !
  integer :: i

  if(n<1) then
    write(6,*) n, __FILE__, __LINE__
    stop 'n<1'
  endif
  
  _dealloc(mat)
  allocate(mat(n,n))
  mat = 0
  
  do i=1,n; mat(i,i) = 1; enddo 
  
end subroutine ! alloc_identity


end module !m_alloc_identity
