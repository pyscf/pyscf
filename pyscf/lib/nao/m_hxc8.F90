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

module m_hxc8

  use m_block_crs8, only : block_crs8_t
  use m_pb_coul_aux, only : pb_coul_aux_t
  
  implicit none

  !! The type will contain the fields for 
  type hxc8_t
    real(4), allocatable :: hxc_pack(:) ! to hold the kernel in packed form
    type(block_crs8_t)   :: bcrs   ! to hold the kernel of overlapping functions
    type(pb_coul_aux_t)  :: ca      ! auxiliary to compute the Hartree kernel
  end type !hxc8_t

end module !m_hxc8
