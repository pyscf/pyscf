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

module m_dp_aux
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_hxc8, only : hxc8_t
  
  implicit none

  type dp_aux_t
    integer :: nspp = -999
    integer :: nc    = -999
    integer :: nc_loc = -999
    integer :: nc_biloc = -999
    real(8) :: coeff_vecs(3,3) = -999
    integer, allocatable :: sp2norbs(:)
    real(8), allocatable :: sp2rcut(:)
    real(8), allocatable :: atom2coord(:,:)
    type(hxc8_t) :: hk
    
  end type ! dp_aux_t
  
contains

!
!
!
subroutine deallocate_dp_aux(a)
  
  implicit none
  type(dp_aux_t), intent(inout) :: a

  _dealloc(a%sp2rcut)
  _dealloc(a%sp2norbs)
  _dealloc(a%atom2coord)
end subroutine

!!
!!
!!
subroutine preinit_dp_aux(sv, a)
  use m_system_vars, only : system_vars_t, get_uc_vecs_ptr, get_sp2norbs, get_sp2rcut
  use m_system_vars, only : get_atom2coord
  use m_bulk_uc_vecs, only : get_coeff_vecs
 
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(dp_aux_t), intent(inout) :: a
  !! internal
  real(8), pointer :: uc_vecs(:,:)
    
  uc_vecs => get_uc_vecs_ptr(sv)
  a%coeff_vecs = get_coeff_vecs(uc_vecs)
  call get_sp2norbs(sv, a%sp2norbs)
  call get_sp2rcut(sv, a%sp2rcut)
  call get_atom2coord(sv, a%atom2coord)
  
  
end subroutine ! preinit_dp_aux
!!

end module !m_dp_aux
