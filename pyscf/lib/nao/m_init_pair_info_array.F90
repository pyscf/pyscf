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

module m_init_pair_info_array

#include "m_define_macro.F90" 
  use m_die, only : die

  implicit none
  private die
 
  contains

!
! Initialization of the pair_info_t in a simplest case, when only specie-indices and coordinates are given.
!
subroutine init_pair_info_array(p2srncc, sv, p2info)
  use m_system_vars, only : system_vars_t
  use iso_c_binding, only: c_double, c_int64_t
  use m_pair_info, only : pair_info_t
  use m_init_pair_info, only : init_pair_info
  
  implicit none
  !! external
  real(c_double), intent(in) :: p2srncc(:,:)
  type(system_vars_t), intent(in) :: sv
  type(pair_info_t), intent(inout), allocatable :: p2info(:)
  !! internal
  integer :: ncc, p, ld, npairs
  
  ld     = ubound(p2srncc,1)
  if(ld<9) _die('ld<9')
  npairs = ubound(p2srncc,2)
  
  _dealloc(p2info)
  allocate(p2info(npairs))

  do p=1,npairs
    ncc = int(p2srncc(9,p))
    if(ncc>0) then 
      call init_pair_info(int(p2srncc(1:2,p)+1), reshape(p2srncc(3:8,p), [3,2]), &
        int(ncc,8), int(p2srncc(10:9+ncc,p)+1,8), sv, p2info(p))
    else
      call init_pair_info(int(p2srncc(1:2,p)+1), reshape(p2srncc(3:8,p), [3,2]), &
        int(ncc,8), [0_8], sv, p2info(p))
    endif
  enddo

end subroutine ! init_pair_info_array

end module !m_init_pair_info_array
